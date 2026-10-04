'use strict';
const $ = id => document.getElementById(id);
const escapeText = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels = {DEPLOY:'部署',RETREAT:'撤退',SKILL:'开技能',SKILL_END:'主动结束技能',SWITCH_MODE:'切换模式'};
const professionNames = {PIONEER:'先锋',WARRIOR:'近卫',TANK:'重装',SNIPER:'狙击',CASTER:'术师',MEDIC:'医疗',SUPPORT:'辅助',SPECIAL:'特种'};
const buildFields = ['char_id','elite','level','trust','potential_rank','skill_id','skill_level','module_id','module_level','auto_skill'];
const freshProject = () => ({format:'arksim-plan-editor',version:1,stage:'',squad:[],operations:[],options:{spawn_timing:'fast',enemy_attack_timing:'float',seed:0,enemy_muzzle:false,enemy_turning:false}});
const state = {project:freshProject(),catalog:null,stage:null,details:new Map(),selectedBuild:null,editIndex:null,tile:null,facing:0,preview:0,dirty:false,draftDirty:false,running:false,revision:0};
let noticeTimer, statTimer, buildRequest=0, stageRequest=0;

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `请求失败 (${response.status})`);
  return value;
}
function notify(message,error=false) {
  clearTimeout(noticeTimer); $('notice').hidden=false; $('notice').textContent=message;
  $('notice').classList.toggle('error',error);
  noticeTimer=setTimeout(()=>$('notice').hidden=true,error?10000:5000);
}
function guard(fn) { return async (...args) => {try {await fn(...args);} catch(error) {notify(error.message,true);}}; }
function markDirty() {
  state.dirty=true; state.revision++; $('saveState').textContent='方案有未保存修改';
  $('openReplay').hidden=true;
}
function operatorName(id) {return state.catalog?.operators.find(o=>o.id===id)?.name || id;}
function frameText(time) {const frame=Math.round(time*30);return `${Math.abs(time*30-frame)>1e-6?'约 ':''}f${frame}`;}
function selectedMember() {return state.project.squad.find(b=>b.char_id===$('opChar').value);}
function detailFor(id) {return state.details.get(id);}
async function loadDetail(id) {
  if (!state.details.has(id)) state.details.set(id,await api(`/api/operator?id=${encodeURIComponent(id)}`));
  return state.details.get(id);
}
function setOptions(select, items, value) {
  select.innerHTML=items.map(item=>`<option value="${escapeText(item.value)}" ${item.disabled?'disabled':''}>${escapeText(item.label)}</option>`).join('');
  if (items.some(item=>String(item.value)===String(value)&&!item.disabled)) select.value=String(value);
}
function unlocked(condition,elite,level) {
  const phase=Number(String(condition?.phase??0).replace('PHASE_',''));
  return elite>phase || (elite===phase&&level>=Number(condition?.level??1));
}
function skillRank(member) {
  const skill=detailFor(member?.char_id)?.skills.find(s=>s.id===member?.skill_id);
  return skill?.ranks.find(r=>r.rank===member.skill_level);
}
function updateStageOptions() {
  const term=$('stageSearch').value.trim().toLowerCase();
  const stages=state.catalog.stages.filter(s=>`${s.id} ${s.code} ${s.name}`.toLowerCase().includes(term)||s.id===state.project.stage);
  setOptions($('stageSelect'),[{value:'',label:'请选择关卡'},...stages.map(s=>({value:s.id,label:`${s.code} · ${s.name}${state.catalog.stages.filter(other=>other.code===s.code&&other.name===s.name).length>1?` · ${s.id}`:''}`}))],state.project.stage);
}
async function selectStage(id, dirty=true) {
  const request=++stageRequest;
  const stage=id?await api(`/api/stage?id=${encodeURIComponent(id)}`):null;
  if (request!==stageRequest) return;
  state.project.stage=id; state.stage=stage; state.tile=null;
  if (dirty) markDirty();
  const meta=state.catalog.stages.find(s=>s.id===id);
  $('stageInfo').textContent=stage?`${meta.code} · ${meta.name} / ${stage.map.width} × ${stage.map.height}`:'尚未选择地图';
  $('mapName').textContent=meta?`${meta.code} · ${meta.name}`:'地图';
  updateStageOptions(); renderMaps(); renderOperationForm();
}
function renderSearch() {
  const term=$('operatorSearch').value.trim().toLowerCase();
  const matches=term?state.catalog.operators.filter(o=>`${o.name} ${o.id}`.toLowerCase().includes(term)):[];
  $('searchResults').innerHTML=matches.slice(0,24).map(o=>`<button class="search-result" data-id="${escapeText(o.id)}"><span>${escapeText(o.name)}</span><small>${escapeText(professionNames[o.profession]||o.profession||'干员')}</small></button>`).join('');
  if (term&&!matches.length) $('searchResults').textContent='没有找到干员';
}
function renderSquad() {
  $('squadCount').textContent=`${state.project.squad.length} 名`;
  $('emptySquad').hidden=state.project.squad.length>0;
  $('squadList').innerHTML=state.project.squad.map(b=>{
    const detail=detailFor(b.char_id),skill=detail?.skills.find(s=>s.id===b.skill_id);
    return `<article class="squad-card ${state.selectedBuild===b.char_id?'selected':''}" data-id="${escapeText(b.char_id)}" tabindex="0" role="button" aria-label="编辑${escapeText(operatorName(b.char_id))}练度"><button class="card-delete" data-remove="${escapeText(b.char_id)}" aria-label="移除${escapeText(operatorName(b.char_id))}">×</button><strong class="name">${escapeText(operatorName(b.char_id))}</strong><span class="meta">E${b.elite} / Lv.${b.level} · ${b.potential_rank+1} 潜</span><div class="skill-line">信赖 ${b.trust}%<br>${escapeText(skill?.name||'不携带技能')}${b.skill_id?` · ${rankName(b.skill_level)}`:''}<br>${escapeText(b.module_id?detail?.modules.find(m=>m.id===b.module_id)?.name||b.module_id:'不装备模组')}${b.module_id?` · ${b.module_level}级`:''}</div></article>`;
  }).join('');
  const before=$('opChar').value;
  setOptions($('opChar'),state.project.squad.map(b=>({value:b.char_id,label:operatorName(b.char_id)})),before);
  $('carriedList').innerHTML=state.project.squad.map(b=>`<button data-char="${escapeText(b.char_id)}" class="${$('opChar').value===b.char_id?'active':''}"><span>${escapeText(operatorName(b.char_id))}</span><small>E${b.elite} / ${b.level}</small></button>`).join('');
  renderOperationForm();
}
function rankName(rank) {return rank>7?`专${['一','二','三'][rank-8]||rank-7}`:`技能 ${rank} 级`;}
function defaultBuild(detail) {
  const elite=detail.maxLevels.length-1,level=detail.maxLevels[elite],trust=200;
  const skill=detail.skills.find(s=>unlocked(s.unlock,elite,level));
  const ranks=skill?.ranks.filter(r=>r.conditions.every(c=>unlocked(c,elite,level)))||[];
  const module=detail.modules.find(m=>unlocked(m.unlock,elite,level)&&m.levels.length);
  return {char_id:detail.id,elite,level,trust,potential_rank:detail.maxPotential,
    skill_id:skill?.id??null,skill_level:ranks.length?Math.max(...ranks.map(r=>r.rank)):null,
    module_id:module?.id??null,module_level:module?Math.max(...module.levels):0,auto_skill:false};
}
async function openBuild(id) {
  clearTimeout(statTimer);
  if (state.draftDirty) await saveMember(false);
  const request=++buildRequest;
  const detail=await loadDetail(id);
  if (request!==buildRequest) return;
  state.selectedBuild=id; state.draftDirty=false;
  $('buildName').textContent=detail.name; $('buildHint').textContent=detail.position==='MELEE'?'部署类型：地面':'部署类型：高台';
  $('buildForm').hidden=false;
  const member=state.project.squad.find(b=>b.char_id===id);
  const build=member??defaultBuild(detail);
  setOptions($('elite'),detail.maxLevels.map((max,i)=>({value:i,label:`精英 ${i}`})),build.elite);
  setOptions($('potential'),Array.from({length:detail.maxPotential+1},(_,i)=>({value:i,label:`${i+1} 潜能`})),build.potential_rank);
  $('level').value=build.level; $('trust').value=build.trust; $('autoSkill').checked=build.auto_skill;
  updateBuildChoices(build);
  $('addMemberBtn').textContent=member?'保存练度':'加入队伍';
  $('buildStats').innerHTML=''; renderSquad(); previewBuild();
}
function updateBuildChoices(preferred=null) {
  const detail=detailFor(state.selectedBuild); if(!detail) return;
  const elite=Number($('elite').value),level=Number($('level').value),trust=Number($('trust').value);
  $('level').max=detail.maxLevels[elite];
  const previousSkill=preferred?(preferred.skill_id??''):$('skill').value;
  const skills=detail.skills.filter(s=>unlocked(s.unlock,elite,level));
  setOptions($('skill'),[{value:'',label:'不携带技能'},...skills.map(s=>({value:s.id,label:s.name}))],previousSkill);
  if(!preferred&&previousSkill&&!skills.some(s=>s.id===previousSkill)) $('skill').value='';
  const skill=skills.find(s=>s.id===$('skill').value);
  const ranks=skill?.ranks.filter(r=>r.conditions.every(c=>unlocked(c,elite,level)))||[];
  setOptions($('skillLevel'),ranks.length?ranks.map(r=>({value:r.rank,label:rankName(r.rank)})):[{value:'',label:'未携带技能'}],preferred?.skill_level??$('skillLevel').value??ranks.at(-1)?.rank);
  if(skill&&!$('skillLevel').value) $('skillLevel').value=String(ranks.at(-1)?.rank??'');
  const modules=detail.modules.filter(m=>unlocked(m.unlock,elite,level));
  const previousModule=preferred?(preferred.module_id??''):$('module').value;
  setOptions($('module'),[{value:'',label:'不装备模组'},...modules.map(m=>({value:m.id,label:m.name}))],previousModule);
  const module=modules.find(m=>m.id===$('module').value);
  setOptions($('moduleLevel'),module?module.levels.map(grade=>({value:grade,label:`${grade} 级${module.trust[String(grade)]>trust?'（信赖不足）':''}`})):[{value:0,label:'未装备模组'}],preferred?.module_level??$('moduleLevel').value);
  $('moduleLevel').disabled=!module; $('skillLevel').disabled=!skill;
}
function readBuild() {
  return {char_id:state.selectedBuild,elite:Number($('elite').value),level:Number($('level').value),trust:Number($('trust').value),potential_rank:Number($('potential').value),skill_id:$('skill').value||null,skill_level:$('skill').value?Number($('skillLevel').value):null,module_id:$('module').value||null,module_level:$('module').value?Number($('moduleLevel').value):0,auto_skill:$('autoSkill').checked};
}
async function previewBuild() {
  const request=++buildRequest; if(!state.selectedBuild) return;
  try {
    const value=await api('/api/build',readBuild()); if(request!==buildRequest) return;
    const a=value.summary.attributes;
    $('buildStats').innerHTML=[['生命',a.maxHp],['攻击',a.atk],['防御',a.def],['费用',a.cost]].map(([label,value])=>`<div><small>${label}</small><b>${Number(value).toFixed(2)}</b></div>`).join('');
  }catch(error){if(request===buildRequest) $('buildStats').textContent=`配置提示：${error.message}`;}
}
async function saveMember(showNotice=true) {
  if(!state.selectedBuild) return;
  const value=await api('/api/build',readBuild());
  const index=state.project.squad.findIndex(b=>b.char_id===value.config.char_id);
  if(index<0) state.project.squad.push(value.config); else state.project.squad[index]=value.config;
  state.draftDirty=false; markDirty(); $('addMemberBtn').textContent='保存练度';
  renderSquad(); renderOperations(); renderMaps();
  if(showNotice) notify('队伍练度已保存到当前方案');
}
function removeMember(id) {
  const refs=state.project.operations.filter(o=>o.char_id===id).length;
  if(refs&&!confirm(`移除${operatorName(id)}会同时移除其 ${refs} 条操作，是否继续？`)) return;
  state.project.squad=state.project.squad.filter(b=>b.char_id!==id);
  state.project.operations=state.project.operations.filter(o=>o.char_id!==id);
  if(state.selectedBuild===id){state.selectedBuild=null;state.draftDirty=false;$('buildForm').hidden=true;$('buildName').textContent='选择一名干员';}
  cancelEdit(); markDirty(); renderSquad(); renderOperations(); renderMaps();
}
function switchStep(step) {
  if(step==='operations'&&(!state.stage||!state.project.squad.length)) {notify('先选择关卡，并将干员加入队伍',true);return;}
  $('setupView').hidden=step!=='setup'; $('operationsView').hidden=step!=='operations';
  document.querySelectorAll('[data-step]').forEach(b=>b.classList.toggle('active',b.dataset.step===step));
  renderSquad(); renderOperations(); renderMaps();
}
function orderedOperations() {return state.project.operations.map((op,index)=>({op,index})).sort((a,b)=>a.op.time-b.op.time||a.index-b.index);}
function plannedUnits() {
  const units=new Map();
  for(const {op} of orderedOperations()) {
    if(op.time>state.preview+1e-9) continue;
    if(op.action==='DEPLOY') units.set(op.char_id,op);
    if(op.action==='RETREAT') units.delete(op.char_id);
  }
  return [...units.values()];
}
function tileClass(tile,allowed=false) {
  return ['tile',tile.height==='HIGHLAND'?'highland':'',tile.build==='NONE'?'none':'',tile.key==='tile_start'?'start':'',tile.key==='tile_end'?'end':'',tile.key==='tile_telin'||tile.key==='tile_telout'?'portal':'',allowed?'allowed':''].join(' ');
}
function renderMaps() {
  const map=state.stage?.map;
  for(const [id,mini] of [['miniMap',true],['mapBoard',false]]) {
    const board=$(id); if(!map){board.innerHTML='';continue;}
    board.style.gridTemplateColumns=`repeat(${map.width},minmax(0,1fr))`;board.style.setProperty('--ratio',`${map.width}/${map.height}`);
    const units=mini?[]:plannedUnits(),member=selectedMember(),position=detailFor(member?.char_id)?.position;
    const tiles=[];
    for(let r=map.height-1;r>=0;r--)for(let c=0;c<map.width;c++) {
      const tile=map.cells[r][c],allowed=!mini&&$('opAction').value==='DEPLOY'&&tile.build===position;
      const names=units.filter(o=>o.tile?.[0]===r&&o.tile?.[1]===c);
      const unit=names.at(-1);
      const door=tile.key==='tile_start'?'RED':tile.key==='tile_end'?'BLUE':tile.key==='tile_telin'?'IN':tile.key==='tile_telout'?'OUT':'';
      const selected=state.tile?.[0]===r&&state.tile?.[1]===c;
      tiles.push(`<button type="button" class="${tileClass(tile,allowed)} ${selected&&!mini?'selected':''}" data-r="${r}" data-c="${c}" title="${r}, ${c} · ${tile.build==='MELEE'?'地面':tile.build==='RANGED'?'高台':'不可部署'}${names.length>1?' · 计划位置冲突':''}" aria-label="地图行${r}列${c}${allowed?'可部署':''}" ${mini?'tabindex="-1"':''}><span class="door ${tile.key==='tile_start'?'red':'blue'}">${door}</span><span class="coordinate">${r},${c}</span>${unit?`<span class="unit"><b>${escapeText(operatorName(unit.char_id))}</b><span>${['→','↓','←','↑'][unit.facing??0]}</span></span>`:''}</button>`);
    }
    board.innerHTML=tiles.join('');
  }
  $('mapClock').textContent=`${state.preview.toFixed(3)} 秒 · ${frameText(state.preview)}`;
}
function renderOperationForm() {
  const action=$('opAction').value,member=selectedMember(),rank=skillRank(member);
  $('deploymentFields').hidden=action!=='DEPLOY';$('modeField').hidden=action!=='SWITCH_MODE';
  $('tileReadout').textContent=state.tile?`行 ${state.tile[0]} · 列 ${state.tile[1]}`:'在地图上点击可部署格';
  document.querySelectorAll('[data-facing]').forEach(b=>b.classList.toggle('active',Number(b.dataset.facing)===state.facing));
  const supported={DEPLOY:true,RETREAT:true,SKILL:rank?.type==='MANUAL',SKILL_END:rank?.controls.can_end,SWITCH_MODE:rank?.controls.can_switch};
  for(const option of $('opAction').options) option.disabled=!supported[option.value];
  $('actionHint').textContent=!supported[action]?'携带技能不支持此操作，请修改操作类型或技能。':action==='DEPLOY'?'点地图选择位置，再选择朝向。':action==='SKILL'?'到达此时刻后尝试开技能；技力不足时按所选策略处理。':action==='SKILL_END'?'只允许主动关闭的技能；结束不会返还技力。':action==='SWITCH_MODE'?'支持切到另一状态或指定模式；切换技力与动画规则仍为候选。':'撤退后再部署仍受冷却与费用限制。';
  $('addOperationBtn').disabled=!member||!supported[action];
}
function setTime(time,writeSeconds=true) {
  if(writeSeconds)$('opTime').value=Number(time.toFixed(9));
  $('opFrame').value=Math.round(time*30);
  state.preview=time;updatePreviewRange();renderMaps();
}
function updatePreviewRange() {
  const end=Math.max(120,Number($('opTime').value)||0,...state.project.operations.map(o=>Number(o.time)||0));
  $('previewTime').max=end+10;$('previewTime').value=state.preview;
}
function operationDraft() {
  const action=$('opAction').value,time=Number($('opTime').value);
  if($('opTime').value===''||!Number.isFinite(time)||time<0) throw new Error('时刻须为非负数');
  const op={char_id:$('opChar').value,time,action,on_failure:$('failure').value};
  if(!op.char_id) throw new Error('请选择携带干员');
  if(action==='DEPLOY') {
    if(!state.tile) throw new Error('请在地图上选择部署格');
    const tile=state.stage.map.cells[state.tile[0]][state.tile[1]],position=detailFor(op.char_id)?.position;
    if(tile.build!==position) throw new Error('此干员不能部署在该格子');
    op.tile=[...state.tile];op.facing=state.facing;
  }
  if(action==='SWITCH_MODE'&&$('opMode').value!=='') op.mode=Number($('opMode').value);
  return op;
}
function addOperation() {
  const op=operationDraft();
  if(state.editIndex===null)state.project.operations.push(op);else state.project.operations[state.editIndex]=op;
  markDirty();cancelEdit();renderOperations();renderMaps();notify('操作已保存到当前方案');
}
function cancelEdit() {
  state.editIndex=null;$('operationTitle').textContent='添加操作';$('addOperationBtn').textContent='＋ 加入操作';$('cancelEditBtn').hidden=true;
}
function editOperation(index) {
  const op=state.project.operations[index];state.editIndex=index;
  $('opChar').value=op.char_id;$('opAction').value=op.action;$('failure').value=op.on_failure||'WAIT';$('opMode').value=op.mode??'';
  state.tile=op.tile?[...op.tile]:null;state.facing=op.facing??0;setTime(op.time);
  $('operationTitle').textContent='编辑操作';$('addOperationBtn').textContent='保存修改';$('cancelEditBtn').hidden=false;
  renderSquad();renderOperations();renderMaps();
}
function moveOperation(index,direction) {
  const rows=orderedOperations(),at=rows.findIndex(x=>x.index===index),other=rows[at+direction];
  if(!other||other.op.time!==rows[at].op.time) return;
  const a=state.project.operations[index];state.project.operations[index]=other.op;state.project.operations[other.index]=a;
  if(state.editIndex===index)state.editIndex=other.index;else if(state.editIndex===other.index)state.editIndex=index;
  markDirty();renderOperations();renderMaps();
}
function renderOperations() {
  const rows=orderedOperations();$('operationCount').textContent=`${rows.length} 条`;updatePreviewRange();
  $('operationList').innerHTML=rows.length?rows.map(({op,index},at)=>{
    const detail=op.action==='DEPLOY'?`(${op.tile?.join(', ')}) · ${['右','下','左','上'][op.facing??0]}`:op.action==='SWITCH_MODE'?`模式 ${op.mode===undefined||op.mode===null?'切到另一状态':op.mode===0?'初始':'技能'}`:'';
    return `<article class="operation-row ${state.editIndex===index?'selected':''}" data-index="${index}" tabindex="0" role="button" aria-label="编辑第${at+1}条操作"><div class="time">${op.time.toFixed(3)}<small>${frameText(op.time)}</small></div><div class="description">${escapeText(operatorName(op.char_id))} · ${escapeText(labels[op.action]||op.action)}<small>${escapeText(detail)} / ${{WAIT:'等待',SKIP:'跳过',STOP:'停止'}[op.on_failure||'WAIT']}</small></div><div class="row-actions"><button data-move="-1" aria-label="同刻上移" ${at===0||rows[at-1].op.time!==op.time?'disabled':''}>↑</button><button data-move="1" aria-label="同刻下移" ${at===rows.length-1||rows[at+1].op.time!==op.time?'disabled':''}>↓</button><button data-delete="1" aria-label="删除此操作">×</button></div></article>`;
  }).join(''):'<div class="empty-state"><h2>还没有操作</h2><p>选干员、定时刻，在地图上部署第一位。</p></div>';
}
function readOptions() {
  state.project.options={spawn_timing:$('spawnTiming').value,enemy_attack_timing:$('attackTiming').value,seed:Number($('seed').value),enemy_muzzle:$('enemyMuzzle').checked,enemy_turning:$('enemyTurning').checked};
}
async function validated() {
  if(state.draftDirty)await saveMember(false);
  readOptions();return api('/api/validate',state.project);
}
function download(url,name) {
  const link=document.createElement('a');link.href=url;link.download=name;
  document.body.append(link);link.click();link.remove();
}
async function saveProject() {
  const compiled=await validated();
  const saved=await api('/api/save',compiled.project);
  state.dirty=false;$('saveState').textContent='方案已保存到本地';
  download(saved.projectUrl,`arksim-${state.project.stage}-方案.json`);
  notify(`方案已保存：${saved.projectFile}。可用“导入”继续编辑。`);
}
async function importFile(file) {
  if((state.dirty||state.draftDirty)&&!confirm('导入将替换当前编辑内容。未保存的修改会丢失，是否继续？')) return;
  const raw=JSON.parse((await file.text()).replace(/^\uFEFF/,''));
  let compiled;
  if(Array.isArray(raw)) {
    if(!state.project.stage)throw new Error('旧操作 JSON 不含地图，请先选择对应关卡，再导入');
    compiled=await api('/api/import-plan',{plan:raw,stage:state.project.stage});
  }else compiled=await api('/api/validate',raw);
  const project=compiled.project;
  await Promise.all(project.squad.map(b=>loadDetail(b.char_id)));
  await selectStage(project.stage,false);
  state.project=project;state.selectedBuild=null;state.draftDirty=false;state.dirty=false;state.revision++;cancelEdit();
  $('buildForm').hidden=true;$('buildName').textContent='选择一名干员';
  const opts=project.options;$('spawnTiming').value=opts.spawn_timing;$('attackTiming').value=opts.enemy_attack_timing;$('seed').value=opts.seed;$('enemyMuzzle').checked=opts.enemy_muzzle;$('enemyTurning').checked=opts.enemy_turning;
  $('saveState').textContent=`已导入 ${file.name}`;$('openReplay').hidden=true;
  updateStageOptions();renderSquad();renderOperations();renderMaps();notify('方案已导入');
}
async function simulate() {
  if(state.running)return;
  const compiled=await validated();
  if(!compiled.plan.length)throw new Error('请先添加至少一条操作');
  const revision=state.revision;
  const job=await api('/api/simulate',state.project);
  state.running=true;$('simulateBtn').disabled=true;$('runStatus').textContent='正在运行主战斗引擎…';$('openReplay').hidden=true;
  const poll=async()=>{
    try {
      const status=await api(`/api/job?job=${encodeURIComponent(job.id)}`);
      if(status.status==='running'){setTimeout(poll,1200);return;}
      state.running=false;$('simulateBtn').disabled=false;
      if(status.status==='failed'){$('runStatus').textContent='模拟失败，请检查运行日志';notify(status.message,true);return;}
      const r=status.result;
      $('runStatus').textContent=`${r.win?'通关':'未通关'} · ${r.enemies_killed} 击杀 / ${r.enemies_leaked} 漏怪 · ${r.time.toFixed(2)} 秒${state.revision!==revision?'（编辑内容已有修改，此回放对应运行时方案）':''}`;
      $('openReplay').href=status.viewerUrl;$('openReplay').hidden=false;
      notify(`模拟完成${r.mechanic_warnings||r.behavior_warnings?'，结果中有未覆盖机制，请检查摘要':''}`);
    }catch(error){state.running=false;$('simulateBtn').disabled=false;$('runStatus').textContent='运行查询失败，请检查本地服务';notify(error.message,true);}
  };
  setTimeout(poll,500);
}

$('stageSearch').addEventListener('input',()=>state.catalog&&updateStageOptions());
$('stageSelect').addEventListener('change',guard(event=>selectStage(event.target.value)));
$('operatorSearch').addEventListener('input',()=>state.catalog&&renderSearch());
$('searchResults').addEventListener('click',guard(event=>{const button=event.target.closest('[data-id]');if(button)return openBuild(button.dataset.id);}));
$('squadList').addEventListener('click',guard(event=>{const remove=event.target.closest('[data-remove]');if(remove)return removeMember(remove.dataset.remove);const card=event.target.closest('[data-id]');if(card)return openBuild(card.dataset.id);}));
$('squadList').addEventListener('keydown',guard(async event=>{if(event.target.classList.contains('squad-card')&&['Enter',' '].includes(event.key)){event.preventDefault();await openBuild(event.target.dataset.id);}}));
$('buildForm').addEventListener('submit',guard(event=>{event.preventDefault();return saveMember();}));
$('buildForm').addEventListener('input',()=>{
  state.draftDirty=true;$('saveState').textContent='练度尚未加入／保存';
  clearTimeout(statTimer);statTimer=setTimeout(previewBuild,300);
});
for(const id of ['elite','level','trust','skill','module']) $(id).addEventListener('change',()=>{
  if(id==='elite')$('level').value=Math.min(Number($('level').value),detailFor(state.selectedBuild).maxLevels[Number($('elite').value)]);
  updateBuildChoices();clearTimeout(statTimer);statTimer=setTimeout(previewBuild,200);
});
document.querySelectorAll('[data-step]').forEach(button=>button.addEventListener('click',guard(async()=>{if(state.draftDirty)await saveMember(false);switchStep(button.dataset.step);})));
$('continueBtn').addEventListener('click',guard(async()=>{if(state.draftDirty)await saveMember(false);switchStep('operations');}));
$('backToSquadBtn').addEventListener('click',()=>switchStep('setup'));
$('opChar').addEventListener('change',()=>{state.tile=null;renderSquad();renderMaps();});
$('opAction').addEventListener('change',()=>{renderOperationForm();renderMaps();});
$('opTime').addEventListener('input',()=>{if($('opTime').value==='')return;const value=Number($('opTime').value);if(Number.isFinite(value)&&value>=0)setTime(value,false);});
$('opFrame').addEventListener('input',()=>{if($('opFrame').value==='')return;const value=Number($('opFrame').value);if(Number.isInteger(value)&&value>=0)setTime(value/30);});
document.querySelectorAll('[data-facing]').forEach(button=>button.addEventListener('click',()=>{state.facing=Number(button.dataset.facing);renderOperationForm();}));
$('mapBoard').addEventListener('click',guard(event=>{
  const button=event.target.closest('[data-r]');if(!button)return;
  if($('opAction').value!=='DEPLOY')return;
  const tile=[Number(button.dataset.r),Number(button.dataset.c)],cell=state.stage.map.cells[tile[0]][tile[1]];
  if(cell.build!==detailFor($('opChar').value)?.position)throw new Error('此干员不能部署在该格子，请选择高亮的可部署格');
  state.tile=tile;renderOperationForm();renderMaps();
}));
$('operationForm').addEventListener('submit',guard(event=>{event.preventDefault();addOperation();}));
$('cancelEditBtn').addEventListener('click',()=>{cancelEdit();renderOperations();});
$('operationList').addEventListener('click',event=>{
  const row=event.target.closest('[data-index]');if(!row)return;const index=Number(row.dataset.index);
  if(event.target.closest('[data-delete]')){state.project.operations.splice(index,1);cancelEdit();markDirty();renderOperations();renderMaps();}
  else if(event.target.closest('[data-move]'))moveOperation(index,Number(event.target.closest('[data-move]').dataset.move));
  else editOperation(index);
});
$('operationList').addEventListener('keydown',event=>{if(event.target.classList.contains('operation-row')&&['Enter',' '].includes(event.key)){event.preventDefault();editOperation(Number(event.target.dataset.index));}});
$('previewTime').addEventListener('input',()=>{state.preview=Number($('previewTime').value);renderMaps();});
$('followTimeBtn').addEventListener('click',()=>{state.preview=Number($('opTime').value)||0;updatePreviewRange();renderMaps();});
for(const id of ['spawnTiming','attackTiming','seed','enemyMuzzle','enemyTurning']) $(id).addEventListener('change',()=>{readOptions();markDirty();});
$('validateBtn').addEventListener('click',guard(async()=>{const result=await validated();notify(`检查通过，共 ${result.plan.length} 条操作。运行时仍会检查费用、冷却与技力。`);}));
$('exportBtn').addEventListener('click',guard(async()=>{const result=await validated();const saved=await api('/api/save',result.project);download(saved.planUrl,`arksim-${state.project.stage}-操作.json`);notify(`操作 JSON 已保存：${saved.planFile}。`);}));
$('saveBtn').addEventListener('click',guard(saveProject));
$('simulateBtn').addEventListener('click',guard(simulate));
$('importBtn').addEventListener('click',()=>$('importFile').click());
$('importFile').addEventListener('change',guard(async event=>{const file=event.target.files?.[0];if(file)try{await importFile(file);}finally{event.target.value='';}}));
window.addEventListener('beforeunload',event=>{if(state.dirty||state.draftDirty){event.preventDefault();event.returnValue='';}});

(async()=>{
  try {
    if(location.protocol==='file:')throw new Error('编辑页需要本地服务读取数据。回放页仍可直接打开。');
    state.catalog=await api('/api/catalog');
    $('connection').textContent=`本地数据已连接 · ${state.catalog.stages.length} 个关卡 / ${state.catalog.operators.length} 名干员 · v${state.catalog.engineVersion}`;
    updateStageOptions();renderSquad();renderOperations();
    if(!state.catalog.stages.length)notify('没有可用地图，请先向本地数据目录导入关卡文件',true);
  }catch(error){$('offline').hidden=false;$('offlineReason').textContent=error.message;$('connection').textContent='未连接本地服务';}
})();
