'use strict';
(() => {
  const el=id=>document.getElementById(id);
  let jobId=null,branch=null,branchReady=false,pendingBranch=null;
  const protocol=location.protocol==='http:'||location.protocol==='https:';
  async function api(path,body){
    const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const value=await response.json();if(!response.ok)throw new Error(value.error||'本地服务未响应');return value;
  }
  function post(pane,type,value={}){pane.contentWindow?.postMessage({arksimWorkspace:true,type,...value},location.origin==='null'?'*':location.origin);}
  function allowLeave(){
    if(branch&&!confirm('返回将保留原回放，但退出当前续编。请先保存需要保留的方案，是否返回？'))return false;
    return true;
  }
  function mode(value){
    mobilePanel('');
    document.body.dataset.mode=value;el('modePicker').hidden=value!=='home';el('plannerPane').hidden=value!=='write';
    el('replayEvents').hidden=value==='branch';el('branchPanel').hidden=value!=='branch';
    el('writeMode').classList.toggle('active',value==='write');el('replayMode').classList.toggle('active',value==='replay'||value==='branch');
    el('cancelBranch').hidden=value!=='branch';el('branchMode').hidden=value!=='replay'||!window.replayController;
    el('timeline').disabled=value==='branch';
    el('caseSelect').disabled=value==='branch';el('closeDataBtn').disabled=value==='branch';
    document.querySelector('.reserve-head span').textContent=value==='branch'?'点击卡片选操作干员':'点击卡片查看 · 回放只读';
    el('mapNote').textContent=value==='branch'?'地图保持切入帧的真实状态；新操作加入方案后，重算后续查看结果。':'点击地图单位或待部署卡片查看状态。按 Space 播放，←/→ 逐帧。';
    el('workspaceHint').textContent=value==='write'?'关卡 · 队伍 · 操作':value==='branch'?`续编 f${branch.frame} · 新操作从 f${branch.frame+1} 开始`:value==='replay'?'回放只读 · 可暂停后续编':'选择工作模式';
    el('dataLoader').hidden=value!=='replay'||!!window.replayController;
    window.replayController?.pause();window.replayController?.redraw();
  }
  async function write(){
    if(!allowLeave())return;
    branch=null;window.replayController?.selection(null);
    if(!protocol){el('workspaceMessage').textContent='写操作需要本地服务：在项目目录运行 python tools/editor.py，再打开 http://127.0.0.1:8873/。离线仍可看回放。';mode('home');return;}
    try{await api('/api/catalog');if(!el('plannerPane').getAttribute('src'))el('plannerPane').src='planner.html?embed=workspace';mode('write');}
    catch{el('workspaceMessage').textContent='本地编辑服务未连接，请启动 python tools/editor.py 后重试。';mode('home');}
  }
  function showReplay(){if(!allowLeave())return;branch=null;window.replayController?.selection(null);mode('replay');}
  function loadReplay(data,id=null,time=0){
    if(!data?.cases?.length||!data.map||data.cases.some(c=>!Array.isArray(c.frames)||!c.frames.length))throw new Error('文件不是有效的回放数据');
    window.replayController?.dispose();jobId=id;branch=null;startReplay(data);mode('replay');
    window.replayController.seekTime(time);el('closeDataBtn').hidden=false;
  }
  async function openJob(id,time=0){const data=await api(`/api/replay?job=${encodeURIComponent(id)}`);loadReplay(data,id,time);}
  async function prepareBranch(project=null){
    const replay=window.replayController;if(!replay)return;
    if(!protocol){el('branchSetup').hidden=false;el('branchMessage').textContent='续编需要本地服务。请启动 python tools/editor.py，在工作台重新打开这份回放。';return;}
    if(el('branchPane').contentWindow?.editorBridge?.isRunning()){el('branchSetup').hidden=false;el('branchMessage').textContent='上一份续编仍在运行，请等完成后再选择新的切入点。';return;}
    replay.pause();el('branchSetup').hidden=false;el('branchProjectLabel').hidden=true;el('branchMessage').textContent='正在准备这一帧的原方案…';
    const current=replay.currentFrame(),frame=current.fixedFrame??Math.round(current.t*30);
    try{
      const payload={frame,case:replay.currentCaseIndex()};
      if(jobId)payload.job=jobId;else{payload.replay=replay.data;if(project)payload.project=project;}
      branch=await api('/api/branch/prepare',payload);
      branch.sourceJob=jobId;pendingBranch=branch;
      if(!el('branchPane').getAttribute('src'))el('branchPane').src='planner.html?embed=branch';
      else if(branchReady){post(el('branchPane'),'branch', {branch});pendingBranch=null;}
      el('branchTitle').textContent=`续编 / ${branch.time.toFixed(3)} 秒 · f${branch.frame}`;
      el('branchSetup').hidden=true;mode('branch');
    }catch(error){
      el('branchMessage').textContent=error.message;
      el('branchProjectLabel').hidden=!error.message.includes('project.json');
    }
  }
  window.workspaceSelectUnit=char=>{if(branch&&char)post(el('branchPane'),'unit',{char});};
  window.workspaceSelectTile=tile=>{if(branch)post(el('branchPane'),'tile',{tile});};
  function mobilePanel(value){document.body.dataset.mobilePanel=value;el('mobileUnit').setAttribute('aria-pressed',String(value==='unit'));el('mobileEvents').setAttribute('aria-pressed',String(value==='events'));}
  window.workspaceInspect=()=>{if(document.body.dataset.mode==='replay'&&window.innerWidth<=650)mobilePanel('unit');};
  el('mobileUnit').onclick=()=>mobilePanel(document.body.dataset.mobilePanel==='unit'?'':'unit');
  el('mobileEvents').onclick=()=>mobilePanel(document.body.dataset.mobilePanel==='events'?'':'events');
  window.addEventListener('message',async event=>{
    if(event.origin!==location.origin||!event.data?.arksimWorkspace)return;
    const regular=event.source===el('plannerPane').contentWindow,continuation=event.source===el('branchPane').contentWindow;
    if(!regular&&!continuation)return;
    const message=event.data;
    if(message.type==='ready'){
      if(continuation){branchReady=true;if(pendingBranch){post(el('branchPane'),'branch',{branch:pendingBranch});pendingBranch=null;}}
    }
    if(message.type==='selection'&&continuation&&branch)window.replayController?.selection(message.selection);
    if(message.type==='replay'){
      try{
        if(message.modified&&!confirm('方案在运行后已有修改。仍打开运行时方案的回放？'))return;
        await openJob(message.job,continuation&&branch?branch.time:0);
      }catch(error){el('workspaceMessage').textContent=error.message;}
    }
    if(message.type==='mode'&&message.mode==='replay')showReplay();
  });
  el('pickWrite').onclick=el('writeMode').onclick=write;
  el('pickReplay').onclick=el('replayMode').onclick=showReplay;
  el('homeMode').onclick=()=>{if(!allowLeave())return;branch=null;mode('home');};
  el('cancelBranch').onclick=()=>{if(!allowLeave())return;branch=null;window.replayController?.selection(null);mode('replay');};
  el('branchMode').onclick=()=>prepareBranch();
  el('closeBranchSetup').onclick=()=>{el('branchSetup').hidden=true;};
  el('branchProjectFile').onchange=async event=>{
    try{const file=event.target.files?.[0];if(file)await prepareBranch(JSON.parse((await file.text()).replace(/^\uFEFF/,'')));}
    catch(error){el('branchMessage').textContent=error.message;}finally{event.target.value='';}
  };
  el('closeDataBtn').onclick=()=>{
    if(!allowLeave())return;window.replayController?.dispose();window.replayController=null;branch=null;jobId=null;
    if(protocol)history.replaceState(null,'',location.pathname);el('dataFile').value='';mode('replay');
  };
  el('dataFile').onchange=async event=>{
    const file=event.target.files?.[0];if(!file)return;
    try{el('dataLoaderStatus').textContent='正在读取数据…';loadReplay(JSON.parse((await file.text()).replace(/^\uFEFF/,'')));el('dataLoaderStatus').textContent='';}
    catch(error){el('dataLoaderStatus').textContent=error.message;event.target.value='';}
  };
  mode('home');
  const params=new URLSearchParams(location.search);
  if(EMBEDDED_DATA)loadReplay(EMBEDDED_DATA);
  else if(params.get('job')&&protocol){mode('replay');openJob(params.get('job')).catch(error=>el('dataLoaderStatus').textContent=error.message);}
  else if(params.get('mode')==='write')write();
  else if(params.get('mode')==='replay')showReplay();
})();
