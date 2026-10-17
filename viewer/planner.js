'use strict';
const $ = id => document.getElementById(id);
const escapeText = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels = {DEPLOY:'部署',RETREAT:'撤退',SKILL:'开技能',SKILL_END:'主动结束技能',SWITCH_MODE:'切换模式'};
const professionNames = {PIONEER:'先锋',WARRIOR:'近卫',TANK:'重装',SNIPER:'狙击',CASTER:'术师',MEDIC:'医疗',SUPPORT:'辅助',SPECIAL:'特种'};
const buildFields = ['char_id','elite','level','trust','potential_rank','skill_id','skill_level','module_id','module_level','auto_skill'];
function deploymentTileMatches(position,build) {
  return (build==='MELEE'||build==='RANGED')&&(position==='ALL'||position===build);
}
const freshProject = () => ({format:'arksim-plan-editor',version:1,stage:'',squad:[],operations:[],options:{spawn_timing:'fast',enemy_attack_timing:'float',seed:0,enemy_muzzle:false,enemy_turning:false}});
const state = {project:freshProject(),catalog:null,stage:null,details:new Map(),selectedBuild:null,editIndex:null,tile:null,facing:0,preview:0,dirty:false,draftDirty:false,operationDirty:false,savedLabel:'未保存方案',running:false,revision:0,branch:null};
const embedded=window.parent!==window;
const branchEmbed=new URLSearchParams(location.search).get('embed')==='branch';
document.body.classList.toggle('embedded',embedded);document.body.classList.toggle('branch-editor',branchEmbed);
function sendWorkspace(type,value={}){if(embedded)window.parent.postMessage({arksimWorkspace:true,type,...value},location.origin);}
function lockedOperation(index){return state.branch&&state.project.operations[index]?.time<=state.branch.time+1e-9;}
let noticeTimer, statTimer, buildRequest=0, stageRequest=0;
const buildInputs=['elite','level','trust','potential','skill','skillLevel','module','moduleLevel','autoSkill'];
const operationInputs=['opChar','opAction','opTime','opFrame','opMode','failure'];
const optionInputs=['spawnTiming','attackTiming','seed','enemyMuzzle','enemyTurning'];
let draftTimer,draftReady=false,restoringDraft=false,recoveryDraft=null,draftRevision=0,savedDraftRevision=0;
let previousDraftSession='',draftSession=crypto.randomUUID().replaceAll('-','');
try {previousDraftSession=sessionStorage.getItem('arksim-editor-draft-session')||'';} catch(error) {}

function readFields(ids) {return Object.fromEntries(ids.map(id=>[id,$(id).type==='checkbox'?$(id).checked:$(id).value]));}
function writeFields(fields,ids) {
  for(const id of ids) if(Object.hasOwn(fields,id)) {
    if($(id).type==='checkbox')$(id).checked=fields[id]===true;else $(id).value=fields[id];
  }
}
function showSaveState(message=null) {
  const pending=[state.draftDirty?'练度尚未加入／保存':'',state.operationDirty?'操作表单尚未加入／保存':''].filter(Boolean);
  $('saveState').textContent=pending.length?pending.join('；'):message??(state.dirty?'方案有未保存修改':state.savedLabel);
}
function scheduleDraft() {
  if(!draftReady||restoringDraft)return;
  recoveryDraft=null;$('draftRecovery').hidden=true;
  draftRevision++;$('draftState').textContent='草稿待保存…';
  clearTimeout(draftTimer);draftTimer=setTimeout(()=>flushDraft(),600);
}
function draftSnapshot() {
  return {version:1,project:state.project,ui:{step:$('operationsView').hidden?'setup':'operations',
    selectedBuild:state.selectedBuild,build:readFields(buildInputs),operation:readFields(operationInputs),options:readFields(optionInputs),
    editIndex:state.editIndex,tile:state.tile,facing:state.facing,preview:state.preview,
    dirty:state.dirty,buildDirty:state.draftDirty,operationDirty:state.operationDirty,
    savedLabel:state.savedLabel,stageSearch:$('stageSearch').value,operatorSearch:$('operatorSearch').value}};
}
async function flushDraft(keepalive=false) {
  clearTimeout(draftTimer);
  if(!draftReady||restoringDraft||draftRevision===savedDraftRevision)return;
  const revision=draftRevision;
  const body=JSON.stringify({session:draftSession,revision,snapshot:draftSnapshot()});
  try {sessionStorage.setItem('arksim-editor-draft-session',draftSession);} catch(error) {}
  $('draftState').textContent='正在保存草稿…';
  try {
    const response=await fetch('/api/draft',{method:'POST',headers:{'Content-Type':'application/json'},body,keepalive});
    const saved=await response.json();if(!response.ok)throw new Error(saved.error||'草稿保存失败');
    savedDraftRevision=Math.max(savedDraftRevision,revision);
    if(revision===draftRevision){$('draftState').textContent=`草稿已保存 · ${new Date(saved.savedAt).toLocaleTimeString('zh-CN')}`;$('draftState').classList.remove('error');}
  }catch(error){
    if(revision===draftRevision){$('draftState').textContent='草稿未写入，请保留页面并检查本地服务';$('draftState').classList.add('error');}
  }
}
async function checkDraft() {
  draftReady=true;
  const revision=draftRevision;
  try {
    const result=await api(`/api/draft?session=${encodeURIComponent(previousDraftSession)}`);
    if(draftRevision!==revision)return;
    recoveryDraft=result.draft;
    if(recoveryDraft){
      const project=recoveryDraft.snapshot.project,stage=state.catalog.stages.find(s=>s.id===project.stage);
      $('draftDescription').textContent=`${stage?`${stage.code} · ${stage.name}`:'尚未选择关卡'} / ${project.squad.length} 名干员 / ${project.operations.length} 条操作 · ${new Date(recoveryDraft.savedAt).toLocaleString('zh-CN')}`;
      $('draftRecovery').hidden=false;$('draftState').textContent='发现本地草稿，可恢复继续编辑';
    }else $('draftState').textContent='编辑后自动保存草稿';
  }catch(error){$('draftState').textContent='无法读取草稿；编辑后将重试保存';$('draftState').classList.add('error');}
}
async function loadProject(project) {
  await Promise.all(project.squad.map(b=>loadDetail(b.char_id)));
  const stage=project.stage?await api(stageUrl(project.stage,project.options?.optional_runes,project.options?.battle_buffs)):null;
  stageRequest++;buildRequest++;
  state.project=project;state.stage=stage;state.selectedBuild=null;state.draftDirty=false;state.tile=null;state.preview=0;
  state.operationDirty=false;cancelEdit();$('buildForm').hidden=true;$('buildName').textContent='选择一名干员';
  const opts=project.options;$('spawnTiming').value=opts.spawn_timing;$('attackTiming').value=opts.enemy_attack_timing;$('seed').value=opts.seed;$('enemyMuzzle').checked=opts.enemy_muzzle;$('enemyTurning').checked=opts.enemy_turning;
  updateStageView();renderSquad();renderOperations();renderMaps();
}
async function restoreDraft() {
  if(!recoveryDraft||restoringDraft)return;
  const snapshot=recoveryDraft.snapshot,ui=snapshot.ui;
  restoringDraft=true;document.querySelector('main').inert=true;$('restoreDraftBtn').disabled=true;
  try {
    await loadProject(snapshot.project);
    $('stageSearch').value=ui.stageSearch||'';$('operatorSearch').value=ui.operatorSearch||'';
    if(ui.selectedBuild){
      await openBuild(ui.selectedBuild);
      writeFields(ui.build,buildInputs);
      updateBuildChoices({skill_id:ui.build.skill||null,skill_level:Number(ui.build.skillLevel),module_id:ui.build.module||null,module_level:Number(ui.build.moduleLevel)});
      writeFields(ui.build,buildInputs);state.draftDirty=ui.buildDirty===true;previewBuild();
    }
    writeFields(ui.options,optionInputs);writeFields(ui.operation,operationInputs);
    state.editIndex=Number.isInteger(ui.editIndex)&&state.project.operations[ui.editIndex]?ui.editIndex:null;
    state.tile=ui.tile;state.facing=ui.facing;state.preview=ui.preview;
    state.dirty=ui.dirty===true;state.operationDirty=ui.operationDirty===true;state.revision++;
    state.savedLabel=ui.savedLabel||(!state.dirty&&!state.draftDirty&&!state.operationDirty?ui.saveState:null)||'已恢复草稿，正式方案尚未保存';
    if(state.editIndex!==null){$('operationTitle').textContent='编辑操作';$('addOperationBtn').textContent='保存修改';$('cancelEditBtn').hidden=false;}
    updateStageOptions();renderSearch();switchStep(ui.step==='operations'&&state.stage&&state.project.squad.length?'operations':'setup');
    showSaveState();$('openReplay').hidden=true;
    recoveryDraft=null;$('draftRecovery').hidden=true;notify('草稿已恢复，尚未加入的表单仍需确认保存');
  }finally{restoringDraft=false;document.querySelector('main').inert=false;$('restoreDraftBtn').disabled=false;}
  scheduleDraft();
}

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
  state.dirty=true; state.revision++; showSaveState();scheduleDraft();
  $('openReplay').hidden=true;
}
function operatorName(id) {return state.catalog?.operators.find(o=>o.id===id)?.name || detailFor(id)?.name || id;}
function frameText(time) {const frame=Math.round(time*30);return `${Math.abs(time*30-frame)>1e-6?'约 ':''}f${frame}`;}
function operationMembers() {
  const members=state.project.squad.flatMap(b=>[b,...(detailFor(b.char_id)?.tokens||[]).filter(t=>t.skill===b.skill_id).map(t=>{
    state.details.set(t.id,{id:t.id,name:t.name,position:t.position,skills:[],modules:[],tokens:[]});
    return {char_id:t.id,owner:b.char_id,token:true};
  })]);
  for(const card of state.stage?.predefinedCards||[]) {
    state.details.set(card.detail.id,card.detail);
    members.push({...card.build,predefined:true});
  }
  return members;
}
function selectedMember() {return operationMembers().find(b=>b.char_id===$('opChar').value);}
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
function stageUrl(id,runes=[],buffs=[]) {
  const query=new URLSearchParams({id});for(const rune of runes||[])query.append('rune',rune);
  for(const buff of buffs||[])query.append('buff',buff);
  return `/api/stage?${query}`;
}
async function selectStage(id, dirty=true) {
  const request=++stageRequest;
  const stage=id?await api(`/api/stage?id=${encodeURIComponent(id)}`):null;
  if (request!==stageRequest) return;
  state.project.stage=id; state.stage=stage; state.tile=null;
  state.project.options.optional_runes=[];
  state.project.options.battle_buffs=[];
  if (dirty) markDirty();
  updateStageView();renderSquad();renderMaps(); renderOperationForm();
}
function updateStageView() {
  const meta=state.catalog.stages.find(s=>s.id===state.project.stage),stage=state.stage;
  $('stageInfo').textContent=stage?`${meta?.code||stage.id} · ${meta?.name||''} / ${stage.map.width} × ${stage.map.height}`:'尚未选择地图';
  $('mapName').textContent=meta?`${meta.code} · ${meta.name}`:'地图';
  updateStageOptions();
  const groups=stage?.runeGroups||[];
  $('runeOptions').hidden=!groups.some(g=>g.length);
  $('runeOptions').innerHTML=groups.map((group,index)=>group.length?`<label>沙盘 ${index+1} 级条件<select data-rune-group="${index}"><option value="">不选择</option>${group.map(r=>`<option value="${escapeText(r.id)}" ${(state.project.options.optional_runes||[]).includes(r.id)?'selected':''}>${escapeText(r.description)}</option>`).join('')}</select></label>`:'').join('');
  const buffs=stage?.battleBuffs,selected=state.project.options.battle_buffs||[];
  $('buffOptions').hidden=!buffs?.items?.length;
  $('buffOptions').innerHTML=buffs?.items?.length?`<p>战备补给：已选 ${selected.length} / ${buffs.limit}</p>${buffs.items.map(b=>`<label style="display:block;margin:8px 0"><input type="checkbox" data-buff="${escapeText(b.id)}" ${selected.includes(b.id)?'checked':''} ${state.branch?'disabled':''}> ${escapeText(b.name)}：${escapeText(b.description)}</label>`).join('')}`:'';
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
  setOptions($('opChar'),operationMembers().map(b=>({value:b.char_id,label:b.token?`${operatorName(b.char_id)} · ${operatorName(b.owner)}`:operatorName(b.char_id)})),before);
  $('carriedList').innerHTML=state.project.squad.map(b=>`<button data-char="${escapeText(b.char_id)}" class="${$('opChar').value===b.char_id?'active':''}"><span>${escapeText(operatorName(b.char_id))}</span><small>E${b.elite} / ${b.level}</small></button>`).join('');
  renderOperationForm();
}
function rankName(rank) {return rank>7?`专${['一','二','三'][rank-8]||rank-7}`:`技能 ${rank} 级`;}
function defaultBuild(detail) {
  const elite=detail.maxLevels.length-1,level=detail.maxLevels[elite],trust=200;
  const skill=detail.skills.find(s=>unlocked(s.unlock,elite,level));
  const ranks=skill?.ranks.filter(r=>r.conditions.every(c=>unlocked(c,elite,level)))||[];
  const modules=detail.modules.filter(m=>unlocked(m.unlock,elite,level)&&m.levels.length);
  const module=modules.find(m=>m.id===detail.defaultModuleId)??modules[0];
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
  scheduleDraft();
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
  const ids=new Set([id,...operationMembers().filter(b=>b.owner===id).map(b=>b.char_id)]);
  const refs=state.project.operations.filter(o=>ids.has(o.char_id)).length;
  if(refs&&!confirm(`移除${operatorName(id)}会同时移除其 ${refs} 条操作，是否继续？`)) return;
  state.project.squad=state.project.squad.filter(b=>b.char_id!==id);
  state.project.operations=state.project.operations.filter(o=>!ids.has(o.char_id));
  if(state.selectedBuild===id){state.selectedBuild=null;state.draftDirty=false;$('buildForm').hidden=true;$('buildName').textContent='选择一名干员';}
  cancelEdit(); markDirty(); renderSquad(); renderOperations(); renderMaps();
}
function switchStep(step) {
  if(state.branch&&step==='setup'){notify('续编保留原关卡、队伍和练度；请在普通写操作模式建立新方案',true);return;}
  if(step==='operations'&&(!state.stage||!state.project.squad.length)) {notify('先选择关卡，并将干员加入队伍',true);return;}
  $('setupView').hidden=step!=='setup'; $('operationsView').hidden=step!=='operations';
  document.querySelectorAll('[data-step]').forEach(b=>b.classList.toggle('active',b.dataset.step===step));
  renderSquad(); renderOperations(); renderMaps();
  scheduleDraft();
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
  if(state.branch){
    const detail=detailFor($('opChar').value),position=detail?.buildablePosition??detail?.position,allowed=[];
    state.stage?.map.cells.forEach((row,r)=>row.forEach((cell,c)=>{if(deploymentTileMatches(position,cell.build))allowed.push([r,c]);}));
    sendWorkspace('selection',{selection:{tile:state.tile,facing:state.facing,char:$('opChar').value,action:$('opAction').value,allowed}});return;
  }
  const map=state.stage?.map;
  for(const [id,mini] of [['miniMap',true],['mapBoard',false]]) {
    const board=$(id); if(!map){board.innerHTML='';continue;}
    board.style.gridTemplateColumns=`repeat(${map.width},minmax(0,1fr))`;board.style.setProperty('--ratio',`${map.width}/${map.height}`);
    const units=mini?[]:plannedUnits(),member=selectedMember(),detail=detailFor(member?.char_id),position=detail?.buildablePosition??detail?.position;
    const tiles=[];
    for(let r=map.height-1;r>=0;r--)for(let c=0;c<map.width;c++) {
      const tile=map.cells[r][c],allowed=!mini&&$('opAction').value==='DEPLOY'&&deploymentTileMatches(position,tile.build);
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
  if(state.branch&&time<=state.branch.time+1e-9)throw new Error(`历史已锁定；新操作须在f${state.branch.frame+1}或之后`);
  const op={char_id:$('opChar').value,time,action,on_failure:$('failure').value};
  if(!op.char_id) throw new Error('请选择携带干员');
  if(action==='DEPLOY') {
    if(!state.tile) throw new Error('请在地图上选择部署格');
    const tile=state.stage.map.cells[state.tile[0]][state.tile[1]],detail=detailFor(op.char_id),position=detail?.buildablePosition??detail?.position;
    if(!deploymentTileMatches(position,tile.build)) throw new Error('此干员不能部署在该格子');
    op.tile=[...state.tile];op.facing=state.facing;
  }
  if(action==='SWITCH_MODE'&&$('opMode').value!=='') op.mode=Number($('opMode').value);
  return op;
}
function addOperation() {
  const op=operationDraft();
  if(state.editIndex===null)state.project.operations.push(op);else state.project.operations[state.editIndex]=op;
  state.operationDirty=false;
  markDirty();cancelEdit();renderOperations();renderMaps();notify('操作已保存到当前方案');
}
function cancelEdit() {
  state.operationDirty=false;
  state.editIndex=null;$('operationTitle').textContent='添加操作';$('addOperationBtn').textContent='＋ 加入操作';$('cancelEditBtn').hidden=true;
}
function editOperation(index) {
  if(lockedOperation(index)){notify('此操作属于切入帧及之前的历史，已锁定',true);return;}
  const op=state.project.operations[index];state.editIndex=index;state.operationDirty=false;
  $('opChar').value=op.char_id;$('opAction').value=op.action;$('failure').value=op.on_failure||'WAIT';$('opMode').value=op.mode??'';
  state.tile=op.tile?[...op.tile]:null;state.facing=op.facing??0;setTime(op.time);
  $('operationTitle').textContent='编辑操作';$('addOperationBtn').textContent='保存修改';$('cancelEditBtn').hidden=false;
  renderSquad();renderOperations();renderMaps();
  showSaveState();scheduleDraft();
}
function moveOperation(index,direction) {
  const rows=orderedOperations(),at=rows.findIndex(x=>x.index===index),other=rows[at+direction];
  if(!other||other.op.time!==rows[at].op.time) return;
  if(lockedOperation(index)||lockedOperation(other.index))return;
  const a=state.project.operations[index];state.project.operations[index]=other.op;state.project.operations[other.index]=a;
  if(state.editIndex===index)state.editIndex=other.index;else if(state.editIndex===other.index)state.editIndex=index;
  markDirty();renderOperations();renderMaps();
}
function renderOperations() {
  const rows=orderedOperations();$('operationCount').textContent=`${rows.length} 条`;updatePreviewRange();
  $('operationList').innerHTML=rows.length?rows.map(({op,index},at)=>{
    const detail=op.action==='DEPLOY'?`(${op.tile?.join(', ')}) · ${['右','下','左','上'][op.facing??0]}`:op.action==='SWITCH_MODE'?`模式 ${op.mode===undefined||op.mode===null?'切到另一状态':op.mode===0?'初始':'技能'}`:'';
    const locked=lockedOperation(index);
    return `<article class="operation-row ${locked?'locked':''} ${state.editIndex===index?'selected':''}" data-index="${index}" tabindex="${locked?-1:0}" role="button" aria-disabled="${!!locked}" aria-label="${locked?'已锁定':'编辑'}第${at+1}条操作"><div class="time">${op.time.toFixed(3)}<small>${frameText(op.time)}</small></div><div class="description">${escapeText(operatorName(op.char_id))} · ${escapeText(labels[op.action]||op.action)}<small>${locked?'历史已锁定 · ':''}${escapeText(detail)} / ${{WAIT:'等待',SKIP:'跳过',STOP:'停止'}[op.on_failure||'WAIT']}</small></div><div class="row-actions"><button data-move="-1" aria-label="同刻上移" ${locked||at===0||rows[at-1].op.time!==op.time?'disabled':''}>↑</button><button data-move="1" aria-label="同刻下移" ${locked||at===rows.length-1||rows[at+1].op.time!==op.time?'disabled':''}>↓</button><button data-delete="1" aria-label="删除此操作" ${locked?'disabled':''}>×</button></div></article>`;
  }).join(''):'<div class="empty-state"><h2>还没有操作</h2><p>选干员、定时刻，在地图上部署第一位。</p></div>';
}
function readOptions() {
  state.project.options={...state.project.options,spawn_timing:$('spawnTiming').value,enemy_attack_timing:$('attackTiming').value,seed:Number($('seed').value),enemy_muzzle:$('enemyMuzzle').checked,enemy_turning:$('enemyTurning').checked};
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
  const revision=state.revision;
  const saved=await api('/api/save',compiled.project);
  if(state.revision===revision){state.dirty=false;state.savedLabel='方案已保存到本地';}
  showSaveState();scheduleDraft();
  download(saved.projectUrl,`arksim-${state.project.stage}-方案.json`);
  notify(`方案已保存：${saved.projectFile}。可用“导入”继续编辑。`);
}
async function importFile(file) {
  if(state.branch)throw new Error('续编不能替换原方案；请返回普通写操作模式导入');
  if((state.dirty||state.draftDirty||state.operationDirty)&&!confirm('导入将替换当前编辑内容。未保存的修改会丢失，是否继续？')) return;
  const raw=JSON.parse((await file.text()).replace(/^\uFEFF/,''));
  let compiled;
  if(Array.isArray(raw)) {
    if(!state.project.stage)throw new Error('旧操作 JSON 不含地图，请先选择对应关卡，再导入');
    compiled=await api('/api/import-plan',{plan:raw,stage:state.project.stage});
  }else compiled=await api('/api/validate',raw);
  await loadProject(compiled.project);
  state.dirty=false;state.revision++;
  state.savedLabel=`已导入 ${file.name}`;showSaveState();$('openReplay').hidden=true;
  updateStageOptions();renderSquad();renderOperations();renderMaps();notify('方案已导入');
  scheduleDraft();
}
async function simulate() {
  if(state.running)return;
  const compiled=await validated();
  if(!compiled.plan.length)throw new Error('请先添加至少一条操作');
  const revision=state.revision;
  const job=await api(state.branch?'/api/branch/simulate':'/api/simulate',state.branch?{branch:state.branch.id,project:compiled.project}:compiled.project);
  state.running=true;$('simulateBtn').disabled=true;$('runStatus').textContent='正在运行主战斗引擎…';$('openReplay').hidden=true;
  const poll=async()=>{
    try {
      const status=await api(`/api/job?job=${encodeURIComponent(job.id)}`);
      if(status.status==='running'){setTimeout(poll,1200);return;}
      state.running=false;$('simulateBtn').disabled=false;
      if(status.status==='failed'){$('runStatus').textContent='模拟失败，请检查运行日志';notify(status.message,true);return;}
      const r=status.result;
      if(state.revision===revision){state.dirty=false;state.savedLabel='运行方案已保存到本地';showSaveState();scheduleDraft();}
      $('runStatus').textContent=`${r.win?'通关':'未通关'} · ${r.enemies_killed} 击杀 / ${r.enemies_leaked} 漏怪 · ${r.time.toFixed(2)} 秒${state.revision!==revision?'（编辑内容已有修改，此回放对应运行时方案）':''}`;
      $('openReplay').href=status.viewerUrl;$('openReplay').hidden=false;
      $('openReplay').onclick=embedded?event=>{event.preventDefault();sendWorkspace('replay',{job:job.id,modified:state.revision!==revision});}:null;
      notify(`模拟完成${r.mechanic_warnings||r.behavior_warnings?'，结果中有未覆盖机制，请检查摘要':''}`);
    }catch(error){state.running=false;$('simulateBtn').disabled=false;$('runStatus').textContent='运行查询失败，请检查本地服务';notify(error.message,true);}
  };
  setTimeout(poll,500);
}

$('stageSearch').addEventListener('input',()=>state.catalog&&updateStageOptions());
$('stageSelect').addEventListener('change',guard(event=>selectStage(event.target.value)));
$('runeOptions').addEventListener('change',guard(async()=>{
  const runes=[...$('runeOptions').querySelectorAll('select')].map(s=>s.value).filter(Boolean);
  const request=++stageRequest,id=state.project.stage;
  const stage=await api(stageUrl(id,runes,state.project.options.battle_buffs));if(request!==stageRequest)return;
  state.project.options.optional_runes=runes;state.stage=stage;state.tile=null;
  markDirty();updateStageView();renderMaps();renderOperationForm();
}));
$('buffOptions').addEventListener('change',guard(async event=>{
  const id=event.target.dataset.buff;if(!id||state.branch)return;
  const previous=state.project.options.battle_buffs||[];
  const buffs=event.target.checked?[...previous,id]:previous.filter(b=>b!==id);
  try {
    const stage=await api(stageUrl(state.project.stage,state.project.options.optional_runes,buffs));
    state.project.options.battle_buffs=buffs;state.stage=stage;markDirty();
  } finally {updateStageView();}
}));
$('operatorSearch').addEventListener('input',()=>state.catalog&&renderSearch());
$('searchResults').addEventListener('click',guard(event=>{const button=event.target.closest('[data-id]');if(button)return openBuild(button.dataset.id);}));
$('squadList').addEventListener('click',guard(event=>{const remove=event.target.closest('[data-remove]');if(remove)return removeMember(remove.dataset.remove);const card=event.target.closest('[data-id]');if(card)return openBuild(card.dataset.id);}));
$('squadList').addEventListener('keydown',guard(async event=>{if(event.target.classList.contains('squad-card')&&['Enter',' '].includes(event.key)){event.preventDefault();await openBuild(event.target.dataset.id);}}));
$('buildForm').addEventListener('submit',guard(event=>{event.preventDefault();return saveMember();}));
$('buildForm').addEventListener('input',()=>{
  state.draftDirty=true;showSaveState();
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
  if(!deploymentTileMatches(detailFor($('opChar').value)?.buildablePosition??detailFor($('opChar').value)?.position,cell.build))throw new Error('此干员不能部署在该格子，请选择高亮的可部署格');
  state.tile=tile;renderOperationForm();renderMaps();
}));
$('operationForm').addEventListener('submit',guard(event=>{event.preventDefault();addOperation();}));
$('cancelEditBtn').addEventListener('click',()=>{cancelEdit();showSaveState();renderOperations();scheduleDraft();});
$('operationList').addEventListener('click',event=>{
  const row=event.target.closest('[data-index]');if(!row)return;const index=Number(row.dataset.index);
  if(lockedOperation(index)){notify('历史操作已锁定',true);return;}
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
function markOperationDraft() {state.operationDirty=true;showSaveState();scheduleDraft();}
for(const event of ['input','change']){
  $('operationForm').addEventListener(event,markOperationDraft);
  document.addEventListener(event,e=>{if(e.target.matches('input,select'))scheduleDraft();});
}
document.querySelectorAll('[data-facing]').forEach(button=>button.addEventListener('click',markOperationDraft));
$('mapBoard').addEventListener('click',event=>{if(event.target.closest('[data-r]')&&state.tile&&$('opAction').value==='DEPLOY')markOperationDraft();});
$('followTimeBtn').addEventListener('click',scheduleDraft);
$('restoreDraftBtn').addEventListener('click',guard(restoreDraft));
$('skipDraftBtn').addEventListener('click',scheduleDraft);
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='hidden')flushDraft(true);});
window.addEventListener('pagehide',()=>flushDraft(true));
window.addEventListener('beforeunload',event=>{flushDraft(true);if(state.dirty||state.draftDirty||state.operationDirty){event.preventDefault();event.returnValue='';}});

(async()=>{
  try {
    if(location.protocol==='file:')throw new Error('编辑页需要本地服务读取数据。回放页仍可直接打开。');
    state.catalog=await api('/api/catalog');
    $('connection').textContent=`本地数据已连接 · ${state.catalog.stages.length} 个关卡 / ${state.catalog.operators.length} 名干员 · v${state.catalog.engineVersion}`;
    updateStageOptions();renderSquad();renderOperations();
    if(!branchEmbed)await checkDraft();else $('draftState').textContent='续编方案可另存；原回放保持不变';
    sendWorkspace('ready');
    if(!state.catalog.stages.length)notify('没有可用地图，请先向本地数据目录导入关卡文件',true);
  }catch(error){$('offline').hidden=false;$('offlineReason').textContent=error.message;$('connection').textContent='未连接本地服务';}
})();

window.editorBridge={isDirty:()=>state.dirty||state.draftDirty||state.operationDirty,isRunning:()=>state.running};
$('returnReplay').onclick=embedded?event=>{event.preventDefault();sendWorkspace('mode',{mode:'replay'});}:null;
window.addEventListener('message',guard(async event=>{
  if(event.source!==window.parent||event.origin!==location.origin||!event.data?.arksimWorkspace||!state.catalog)return;
  const message=event.data;
  if(message.type==='branch'){
    if(state.running)throw new Error('请等待当前续编运行完成');
    await loadProject(message.branch.project);state.branch=message.branch;state.dirty=false;state.revision++;
    $('openReplay').hidden=true;$('openReplay').onclick=null;
    $('importBtn').hidden=true;$('returnReplay').hidden=true;$('backToSquadBtn').hidden=true;
    document.querySelectorAll('[data-step]').forEach(button=>button.hidden=true);
    $('opTime').min=(state.branch.frame+1)/30;$('opFrame').min=state.branch.frame+1;
    setTime((state.branch.frame+1)/30);switchStep('operations');
    $('simulateBtn').textContent='重算后续';$('runStatus').textContent='已保留历史；后续操作可增删、改时刻和同刻顺序。';
    state.savedLabel=`续编 f${state.branch.frame} · 尚未保存`;showSaveState();
  }
  if(message.type==='unit'&&state.branch&&state.project.squad.some(b=>b.char_id===message.char)){
    $('opChar').value=message.char;state.tile=null;renderSquad();renderMaps();
  }
  if(message.type==='tile'&&state.branch&&$('opAction').value==='DEPLOY'){
    const [r,c]=message.tile,cell=state.stage?.map.cells[r]?.[c];
    if(!deploymentTileMatches(detailFor($('opChar').value)?.buildablePosition??detailFor($('opChar').value)?.position,cell?.build))return;
    state.tile=[r,c];renderOperationForm();renderMaps();markOperationDraft();
  }
}));
