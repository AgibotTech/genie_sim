'use strict';
const $ = id => document.getElementById(id);
const statusNames = {received:'Observation received',waiting_model:'Waiting for model',model_returned:'Model returned',planned:'Planned',issued:'Issued',rejected:'Replan needed',episode_done:'Environment terminated',failed:'Request failed',response_recorded:'Response recorded'};
const kindNames = {current:'Current measured point',target:'Planned target',issued:'Issued endpoint',candidate:'Candidate target'};
const cameraNames = {head:'Head camera',hand_left:'Left-hand camera',hand_right:'Right-hand camera'};
const state = {episode:null,request:null,follow:true,offset:0,tab:'overview',detail:null,view:-1,camera:null,arm:'all',layers:new Set(['current','target','issued']),signature:null,catalog:null};
const storageKey = location.pathname+'observer-v2';
try{const saved=JSON.parse(sessionStorage.getItem(storageKey)||'null');if(saved)for(const key of ['episode','request','follow','offset','tab','view','camera','arm'])if(key in saved)state[key]=saved[key];}catch(error){}
function remember(){try{sessionStorage.setItem(storageKey,JSON.stringify({episode:state.episode,request:state.request,follow:state.follow,offset:state.offset,tab:state.tab,view:state.view,camera:state.camera,arm:state.arm}));}catch(error){}}
let selectionVersion=0, renderVersion=0;
const rawCache = new Map();
function node(tag,text,className){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n;}
function json(value){return JSON.stringify(value,null,2);}
function list(value){return Array.isArray(value)?value:[];}
function parse(value){if(typeof value==='string'){try{return JSON.parse(value);}catch(error){}}return value;}
function time(value){return value?new Date(value).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',second:'2-digit'}):'';}
async function api(route,params={}){const response=await fetch('api/'+route+'?'+new URLSearchParams(params),{cache:'no-store'});if(!response.ok)throw new Error('Records are temporarily unavailable; please retry later');return response.json();}
async function artifact(record,requestId=state.request){const key=requestId+'/'+record.name;if(rawCache.has(key))return rawCache.get(key);const value=await api('artifact',{id:requestId,name:record.name});rawCache.set(key,value);if(rawCache.size>40)rawCache.delete(rawCache.keys().next().value);return value;}
function selectOption(select,value,label){const o=node('option',label);o.value=value;select.append(o);}
function currentView(){const views=state.detail?.views||[];return views[Math.min(state.view,views.length-1)];}
function decision(){const v=currentView();return v?.decision||v?.execution_decision||{};}
function scoped(items){const attempt=currentView()?.attempt;return list(items).filter(item=>attempt==null||item.planning_attempt==null||item.planning_attempt===attempt);}
function reports(){return scoped(state.detail.diagnostics).map(d=>d.report).filter(r=>r&&typeof r==='object');}
function chooseRequest(id){selectionVersion++;state.follow=false;state.request=id;state.signature=null;state.view=-1;refresh();}
function gotoTab(tab){state.tab=tab;render();}
function link(parent,title,tab){const b=node('button',title,'text-link');b.onclick=()=>gotoTab(tab);parent.append(b);}
function card(parent,title,index){const box=node('section',undefined,'card'),head=node('div',undefined,'card-heading'),h=node('h2');if(index)h.append(node('span',index,'card-index'));h.append(document.createTextNode(title));head.append(h);box.append(head);parent.append(box);return box;}
function empty(parent,text){parent.append(node('p',text,'empty-inline'));}
function prose(parent,text){if(typeof text==='string'&&text.trim())parent.append(node('p',text,'prose'));}
function disclosure(parent,label,content){const d=node('details');d.append(node('summary',label));if(content instanceof Node)d.append(content);else d.append(node('pre',json(content)));parent.append(d);return d;}
function updateCatalog(data){
 state.catalog=data;$('run').textContent=data.run;$('run').title=data.run;
 const select=$('episodes'),before=select.value;
 const groupSignature=json(data.episodes.map(e=>[e.id,e.count]));
 if(select.dataset.signature!==groupSignature){select.replaceChildren();data.episodes.forEach(e=>selectOption(select,e.id,`${e.task??'Unknown task'} · Episode ${e.episode??'Not provided'} · ${time(e.started)} · ${e.session.slice(0,6)} / ${e.id.slice(0,5)}`));select.dataset.signature=groupSignature;}
 select.value=data.selected_episode||before;state.episode=data.selected_episode;
 if(state.follow||!state.request)state.request=data.requests[0]?.id||null;
 const requests=$('requests'),signature=json(data.requests.map(r=>[r.id,r.status,r.instruction]))+state.request;
 if(requests.dataset.signature!==signature){requests.replaceChildren();requests.dataset.signature=signature;
  for(const row of data.requests){const b=node('button'),top=node('div',undefined,'round-top'),label=node('span');label.append(node('span',undefined,'round-dot '+row.status),document.createTextNode(`Round ${row.sequence??'?'}`));top.append(label,node('span',time(row.started),'round-time'));b.append(top,node('small',row.instruction||statusNames[row.status]||row.status));b.setAttribute('aria-label',`Round ${row.sequence??'?'} · ${statusNames[row.status]||row.status}`);b.setAttribute('aria-pressed',String(row.id===state.request));b.title=row.instruction;b.onclick=()=>chooseRequest(row.id);requests.append(b);}}
 $('round-count').textContent=data.total+' rounds';$('newer').disabled=state.offset===0;$('older').disabled=state.offset+100>=data.total;
 $('page').textContent=data.total?`${state.offset+1}–${Math.min(state.offset+100,data.total)} / ${data.total}`:'0';
 $('follow').textContent=state.follow?'● Live · auto-updating':'◷ Reviewing history · new records still arriving';
 $('connection').textContent=(data.catching_up?'● Reading records':'● Observer connected')+(data.malformed_events?` · ${data.malformed_events} corrupted records`:'');
}
const fieldNames={instruction:'Task instruction',reason:'Decision reason',success_condition:'Success condition',mode:'Execution mode',status:'Status',frame:'Frame',camera:'Camera',sequence:'Current round',planning_attempt:'Planning attempt',task:'Task',task_name:'Task name',episode_idx:'Episode',states:'Robot state',context:'Task context',observation:'Current observation',history:'History',memory:'Memory',task_evaluation:'Task evaluation',execution:'Execution constraints',robot:'Robot parameters',end_pose:'End pose',position:'Position',orientation:'Orientation',left_arm:'Left arm',right_arm:'Right arm',left_gripper:'Left gripper',right_gripper:'Right gripper',motion:'Motion target',waypoints:'Waypoints',candidates:'Candidate targets',tools:'Available tools',arguments:'Arguments',result:'Result',expected_effect:'Expected effect',verification:'Follow-up check',uncertainties:'Uncertainties',depth_enabled:'Depth input',rgb_use:'RGB usage',depth_use:'Depth usage',error:'Error',checks:'Checks',gripper_command_error:'Gripper command error',eef_observed_delta:'Measured end-effector change',issued_steps:'Issued steps',generated_steps:'Generated steps',discarded_steps:'Discarded steps',duration_s:'Duration (s)',previous_sequence:'Action round',pose_reference:'Pose convention',grasp_geometry:'Gripper geometry',gripper_command_reference:'Gripper command convention',input_manifest:'Input manifest',robot_type:'Robot type',calibration:'Camera calibration',depth:'Depth info',image_role:'Image role'};
function simple(value){if(value===true)return 'Yes';if(value===false)return 'No';if(value==null)return 'Not provided';return String(value);}
function readable(parent,value,depth=0){
 value=parse(value);
 if(value==null){prose(parent,'Not provided');return;}
 if(typeof value!=='object'){prose(parent,simple(value));return;}
 if(Array.isArray(value)){if(value.every(v=>v===null||typeof v!=='object')){prose(parent,value.map(simple).join(' · '));return;}value.forEach((v,i)=>{const d=disclosure(parent,`Item ${i+1}`,node('div'));readable(d.lastChild,v,depth+1);});return;}
 const dl=node('dl',undefined,'readable-fields');parent.append(dl);
 for(const [key,v] of Object.entries(value)){if(v&&typeof v==='object'){const d=disclosure(dl,fieldNames[key]||key,node('div'));if(depth<5)readable(d.lastChild,v,depth+1);else d.lastChild.append(node('pre',json(v)));}else{if(typeof v==='string'&&v.length>240){const d=disclosure(dl,fieldNames[key]||key,node('div'));prose(d.lastChild,v);continue;}const row=node('div',undefined,'field-row');row.append(node('dt',fieldNames[key]||key),node('dd',simple(v)));dl.append(row);}}
}
function selectedArtifacts(){const data=state.detail,view=currentView();if(!view||view.attempt==null)return data.artifacts;
 const start=parseInt(view.id,10),next=data.views[state.view+1],end=next?parseInt(next.id,10):Infinity;
 return data.artifacts.filter(r=>parseInt(r.name,10)>=start&&parseInt(r.name,10)<end);
}
function render(){
 const data=state.detail;if(!data)return;renderVersion++;remember();
 $('empty').hidden=true;$('detail').hidden=false;
 const row=data.request,view=currentView(),d=decision();
 $('title').textContent=`Round ${row.sequence??'?'} decision`;
 $('identity').textContent=`${data.episode.task??'Unknown task'} / Episode ${data.episode.episode??'Not provided'} · ${time(row.started)}`;
 $('identity').title=row.id+' · '+row.started;
 $('round-status').textContent=statusNames[row.status]||row.status;$('round-status').className='status-pill '+row.status;
 $('warnings').replaceChildren(...data.warnings.map(w=>node('p',w,'notice')));
 $('task-brief').replaceChildren();if(data.episode.instruction){const p=node('p',undefined,'task-brief');p.append(node('span','Overall task'),document.createTextNode(data.episode.instruction));$('task-brief').append(p);}
 document.querySelectorAll('[data-tab]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.tab===state.tab)));
 const panel=$('panel');panel.replaceChildren();
 if(data.views.length>1){const label=node('label','Plan version','attempt-bar'),select=node('select');select.setAttribute('aria-label','Select planning attempt');data.views.forEach((v,i)=>selectOption(select,String(i),`${i===0?'First plan':'Replan '+i}${i===data.views.length-1?' · latest':''}${v.issued||v.issued_event?' · issued':v.decision?' · target returned':' · no executable decision'}`));select.value=String(state.view);select.onchange=()=>{state.view=Number(select.value);state.camera=null;render();};label.append(select);panel.append(label);}
 if(view&&state.view<data.views.length-1)panel.append(node('p','Viewing an earlier plan version; the final action of this round came from a later plan.','notice attempt-warning'));
 if(state.tab==='overview')renderOverview(panel,view,d);
 else if(state.tab==='input')renderInput(panel,view);
 else if(state.tab==='reason')renderReason(panel,d);
 else if(state.tab==='action')renderAction(panel,view);
 else if(state.tab==='feedback')renderFeedback(panel);
 else renderDebug(panel);
}
function stats(parent,items){const grid=node('div',undefined,'stats');for(const [label,value]of items){const item=node('div',undefined,'stat');item.append(node('strong',String(value)),node('span',label));grid.append(item);}parent.append(grid);}
function renderOverview(panel,view,d){
 const grid=node('div',undefined,'overview-grid'),left=node('div',undefined,'stack'),right=node('div',undefined,'stack');grid.append(left,right);panel.append(grid);
 const visual=card(left,'Scene and target','01');if(view)renderRGB(visual,view,true);else empty(visual,'Waiting for camera frames');
 const goal=card(right,'What this step does','02');goal.append(node('p',d.instruction||'No executable target returned yet','card-title'));
 const issued=view?.issued_event;
 if(issued)stats(goal,[['Issued actions',issued.issued_steps+' steps'],['Generated actions',issued.generated_steps+' steps'],['Execution mode',({motion:'Motion target',policy:'Action policy'})[issued.mode||d.mode]||issued.mode||d.mode||'—']]);
 else goal.append(node('p',d.mode?`Execution mode: ${d.mode}`:'The model has not produced an executable decision yet.','hint'));
 if(d.success_condition){goal.append(node('h3','Expected resulting state'));prose(goal,d.success_condition);}
 const reason=card(right,'Why','03');
 if(d.reason){prose(reason,d.reason);}else empty(reason,'This plan version returned no decision reason.');
 link(reason,'View model evidence and tool results →','reason');
 const trail=card(left,'What happened this round');renderTimeline(trail);
 const feedback=card(right,'Post-execution observation','04');
 if(state.detail.feedback.length){prose(feedback,`${state.detail.feedback.length} follow-up observation(s) available to check what the robot actually did.`);link(feedback,'View follow-up feedback →','feedback');}
 else empty(feedback,issued?'Action issued; the next observation has not arrived yet.':'No follow-up feedback for this round\'s action yet.');
}
function renderTimeline(parent){
 const events=state.detail.events,steps=node('div',undefined,'step-list');
 const add=(title,text,kind='')=>{const item=node('div',undefined,'step '+kind);item.append(node('strong',title));if(text)item.append(node('p',text));steps.append(item);};
 add('Camera frames and robot state received',time(state.detail.request.started));
 for(const e of events){if(e.status==='rejected')add(`Planning attempt ${(e.attempt??0)+1} failed`,e.error==='geometry tool budget exceeded'?'Tool query budget exceeded; replanning':e.error||'Execution constraint check failed','warn');
 else if(e.status==='planned')add('Executable plan generated',e.decision?.instruction,'good');
 else if(e.status==='issued')add(`Issued ${e.issued_steps} steps`,`Generated ${e.generated_steps} steps · discarded ${e.discarded_steps??0}`,'good');
 else if(e.status==='episode_done')add('Environment ended this episode','Task success is decided by the environment evaluation');
 else if(e.exception_type&&e.status!=='phase_finished')add('Request failed',e.exception_type,'warn');}
 if(state.detail.request.status==='waiting_model')add('Waiting for model reply','This status comes from the latest snapshot');
 parent.append(steps);
}
function summaryTexts(value){if(typeof value==='string')return value.trim()?[value]:[];if(Array.isArray(value))return value.flatMap(summaryTexts);if(value&&typeof value==='object')return summaryTexts(value.text??value.summary??value.content??[]);return [];}
function renderReason(panel,d){
 const grid=node('div',undefined,'overview-grid'),left=node('div',undefined,'stack'),right=node('div',undefined,'stack');grid.append(left,right);panel.append(grid);
 const reason=card(left,'Model decision reason');if(d.reason)prose(reason,d.reason);else empty(reason,'This plan version returned no decision reason.');
 const summaries=card(right,'Public reasoning summary');const texts=scoped(state.detail.summaries).flatMap(s=>summaryTexts(s.summary??s.reasoning_summary));
 if(texts.length)texts.forEach(t=>prose(summaries,t));else empty(summaries,'The API returned no public reasoning summary. See the decision reason and the evidence cited by the model.');
 summaries.append(node('p','Shows only what the API actually returned; no internal chain of thought is reconstructed.','hint'));
 const rs=reports();const evidence=card(left,'Evidence cited by the model');evidence.append(node('p','The following is self-reported by the model; source citations are not independently verified.','muted'));
 const entries=rs.flatMap(r=>list(r.evidence));if(!entries.length)empty(evidence,'No evidence report was recorded.');
 entries.forEach(e=>{const item=node('div',undefined,'evidence'),source=node('p',undefined,'source');source.append(node('span',({rgb:'Image',depth:'Depth',tool:'Tool',robot_state:'Robot state'})[e.modality]||e.modality||'Evidence','tag'),document.createTextNode(e.source||''));item.append(source,node('p',e.finding||''));evidence.append(item);});
 const expected=card(right,'What should be seen next');let hasExpected=false;for(const r of rs){if(r.expected_effect){prose(expected,r.expected_effect);hasExpected=true;}if(r.verification){expected.append(node('h3','Next-frame check'));prose(expected,r.verification);hasExpected=true;}for(const u of list(r.uncertainties)){expected.append(node('p',u,'notice'));}}
 if(!hasExpected)empty(expected,d.success_condition||'The model gave no follow-up check.');
 const tools=card(right,'Tool queries');loadTools(tools);
 const estimates=rs.flatMap(r=>list(r.estimates));if(estimates.length){const box=card(left,'Values reported by the model');box.append(node('p','These values come from the diagnostic report and are not independent measurements.','muted'));table(box,['Quantity','Value','Frame / basis'],estimates.map(e=>[e.quantity,`${e.value??'—'} ${e.units||''}`,`${e.frame||''} · ${e.basis||''}`]));}
}
function table(parent,headers,rows){const wrap=node('div',undefined,'table-wrap'),t=node('table'),head=node('thead'),tr=node('tr');headers.forEach(h=>tr.append(node('th',h)));head.append(tr);const body=node('tbody');rows.forEach(values=>{const r=node('tr');values.forEach(v=>r.append(node('td',simple(v))));body.append(r);});t.append(head,body);wrap.append(t);parent.append(wrap);}
function gallery(parent,view){const grid=node('div',undefined,'three-col');for(const im of view?.images||[]){const f=node('figure',undefined,'camera-tile'),img=node('img');img.src='api/image?'+new URLSearchParams({path:im.path});img.alt=(cameraNames[im.camera]||im.camera)+' input this round';img.loading='lazy';const cap=node('figcaption');cap.append(node('span',cameraNames[im.camera]||im.camera),node('span',im.size.join(' × ')));f.append(img,cap);grid.append(f);}parent.append(grid);if(!view?.images.length)empty(parent,'No readable RGB images this round.');}
function renderInput(panel,view){
 const images=card(panel,'Frames seen by the model');images.append(node('p',view?.image_source==='model_input'?'RGB images actually sent for this plan.':'Raw observation of this round; older records did not store the model image input separately.','muted'));gallery(images,view);
 const body=node('div',undefined,'section-space');panel.append(body);const requestId=state.request,version=renderVersion;
 const records=selectedArtifacts(),record=records.find(r=>r.kind==='llm-request');
 const box=card(body,'Text sent to the model');box.append(node('p','Loading the actual request…','muted'));
 if(!record){box.lastChild.remove();empty(box,'No model request recorded.');return;}
 artifact(record,requestId).then(value=>{if(version!==renderVersion)return;box.lastChild.remove();renderMessages(box,value);}).catch(error=>{if(version===renderVersion)box.lastChild.textContent=error.message;});
 const tools=card(body,'Tool round trips');tools.classList.add('section-space');loadTools(tools);
}
function extractMessages(value){
 const body=value.body||value,items=[];
 const push=(role,content)=>{if(content!=null)items.push({role,content});};
 if(body.systemInstruction)push('System instructions',body.systemInstruction.parts||body.systemInstruction);
 if(body.instructions)push('System instructions',body.instructions);
 if(body.developerInstructions)push('Developer instructions',body.developerInstructions);
 const source=body.messages||body.contents||body.input;
 if(typeof source==='string')push('Task input',source);
 else if(Array.isArray(source))for(const item of source){const role=({system:'System instructions',developer:'Developer instructions',user:'User input',assistant:'Previous model replies',model:'Previous model replies',tool:'Tool results'})[item.role]||'Input';push(role,item.content??item.parts??item);}
 if(!items.length)push('Request content',body);
 return items;
}
function renderMessages(parent,value){
 const messages=extractMessages(value);
 messages.forEach((message,i)=>{const d=node('details',undefined,'message');d.append(node('summary',`${message.role} · ${i+1}`));const content=node('div');d.append(content);d.open=i===messages.length-1;renderContent(content,message.content);parent.append(d);});
 parent.append(node('p','Shown in the original request order; English text is exactly what the model received. Full protocol data is under “Debug”.','hint'));
}
function renderContent(parent,value){
 if(typeof value==='string'){
  const decoded=parse(value);
  if(decoded&&typeof decoded==='object'&&decoded.camera){const label=(cameraNames[decoded.camera]||decoded.camera)+(decoded.calibration?' · camera calibration':decoded.depth?' · depth info':' · image info');const d=disclosure(parent,label,node('div'));readable(d.lastChild,decoded);}
  else if(decoded!==value)readable(parent,decoded);else prose(parent,value);return;
 }
 if(Array.isArray(value)){value.forEach(v=>renderContent(parent,v));return;}
 if(!value||typeof value!=='object')return;
 if(typeof value.text==='string'){renderContent(parent,value.text);return;}
 if(value.inlineData||value.inline_data||value.image_url||value.type==='image'||value.type==='input_image'){parent.append(node('p','▧ Image attachment · current RGB shown above; full attachment under Debug','muted'));return;}
 readable(parent,value);
}
async function loadTools(parent){
 const records=selectedArtifacts().filter(r=>/tool-result$/.test(r.kind));
 if(!records.length){empty(parent,'No tool results were recorded for this plan.');return;}
 const version=renderVersion,requestId=state.request,loading=node('p','Loading tool queries…','muted');parent.append(loading);
 try{const values=await Promise.all(records.map(r=>artifact(r,requestId)));if(version!==renderVersion)return;loading.remove();
 values.forEach((v,i)=>{const label=({read_depth_pixels:'Read target pixel depth',match_rgb_patch:'Match before/after frames',locate_static_rgb_patch:'Estimate static target position'})[v.name]||v.name||'Tool queries';const args=parse(v.arguments),result=parse(v.result);
 const d=disclosure(parent,`${i+1}. ${label}${result?.camera?' · '+(cameraNames[result.camera]||result.camera):''}`,node('div'));d.classList.add('tool-card');const body=d.lastChild;
 if(result?.pixels){table(body,['Pixel (u, v)','Model label','Depth'],list(result.pixels).map(p=>[`${p.u}, ${p.v}`,p.label||'—',p.valid?`${p.depth_mm} mm`:'no valid depth']));body.append(node('p','Pixels are chosen by the model; a depth value does not prove the pixel belongs to the target object.','hint'));}
 else readable(body,result);
 const params=disclosure(body,'View arguments',node('div'));readable(params.lastChild,args);
 });
 }catch(error){if(version===renderVersion)loading.textContent=error.message;}
}
function renderAction(panel,view){
 if(!view){empty(panel,'Waiting for observation and target records.');return;}
 const grid=node('div',undefined,'overview-grid'),left=node('div',undefined,'stack'),right=node('div',undefined,'stack');grid.append(left,right);panel.append(grid);
 const image=card(left,'Execution target overlay');renderRGB(image,view,true);
 const target=card(right,'Target of the selected plan');prose(target,decision().instruction||'This plan has no target');
 const points=list(view.points).filter(p=>p.kind!=='candidate');if(points.length)table(target,['Arm / type','Position x, y, z (m)'],points.map(p=>[`${p.arm==='left'?'Left arm':'Right arm'} · ${kindNames[p.kind]}`,p.pose.slice(0,3).map(n=>Number(n).toFixed(3)).join(', ')]));
 target.append(node('p',`Frame base_link · ${view.reference==='gripper_center'?'gripper center':'wrist'} reference point. The issued endpoint is the commanded position.`,'hint'));
 const flow=card(right,'Planning and issuing');renderTimeline(flow);
}
function renderFeedback(panel){
 const box=card(panel,'What the robot did after this round\'s action');
 if(!state.detail.feedback.length){empty(box,'No follow-up observation for this round\'s action yet.');return;}
 state.detail.feedback.forEach(f=>{const data=f.data;const block=node('div',undefined,'section-space');box.append(block);block.append(node('h3',`Feedback on round ${data.previous_sequence} action`));
 const fields={};for(const [key,value]of Object.entries(data))if(!['episode','sequence','status','previous_sequence','playback_status'].includes(key))fields[key]=value;readable(block,fields);});
 box.append(node('p','Feedback reflects measured robot change; whether the object is grasped or the task succeeded is decided by the environment evaluation.','hint'));
}
function renderDebug(panel){
 const intro=card(panel,'Raw record');intro.append(node('p','For troubleshooting protocol or coordinate issues. For everyday observation go back to “Overview”.','muted'));readable(intro,{run:state.catalog.run,request:state.request,recorded_at:state.detail.request.started});
 const raw=node('div',undefined,'section-space');panel.append(raw);artifactDetails(raw,state.detail.artifacts);
}
function debugImages(parent,value,requestId){
 const refs=new Set();
 function walk(v){if(!v||typeof v!=='object')return;if(typeof v.$artifact==='string'&&/\.(jpg|png)$/.test(v.$artifact))refs.add(v.$artifact);Object.values(v).forEach(walk);}
 walk(value);if(!refs.size)return;
 const gallery=disclosure(parent,`Image attachments · ${refs.size}`,node('div'));
 for(const ref of refs){const f=node('figure'),img=node('img');img.src='api/image?'+new URLSearchParams({path:requestId+'/'+ref});img.alt='Archived image '+ref;img.loading='lazy';img.style.maxWidth='100%';f.append(img,node('figcaption',ref));gallery.lastChild.append(f);}
}
function artifactDetails(parent,records){
 for(const record of records){const d=disclosure(parent,record.name,node('div')),output=d.lastChild;let loaded=false;const requestId=state.request;
 d.addEventListener('toggle',async()=>{if(!d.open||loaded)return;loaded=true;output.textContent='Loading…';try{const value=await artifact(record,requestId);output.replaceChildren();debugImages(output,value,requestId);output.append(node('pre',json(value)));}catch(error){output.textContent=error.message;loaded=false;}});}
}
function renderRGB(panel,view,overlay){
 if(!view.images.length){panel.append(node('p','No RGB attachments to show this round.','notice'));return;}
 const controls=node('div',undefined,'controls');const label=node('label','Camera '),camera=node('select');camera.setAttribute('aria-label','Select RGB camera');view.images.forEach((im,i)=>selectOption(camera,String(i),cameraNames[im.camera]||im.camera));
 let imageIndex=view.images.findIndex(im=>im.camera===state.camera);if(imageIndex<0)imageIndex=0;camera.value=String(imageIndex);state.camera=view.images[imageIndex].camera;
 camera.onchange=()=>{state.camera=view.images[Number(camera.value)].camera;render();};label.append(camera);controls.append(label);
 if(overlay){const arms=node('select');arms.setAttribute('aria-label','Select arm');[['all','Both arms'],['left','Left arm'],['right','Right arm']].forEach(([v,l])=>selectOption(arms,v,l));arms.value=state.arm;arms.onchange=()=>{state.arm=arms.value;render();};controls.append(arms);}
 panel.append(controls);const image=view.images[imageIndex];
 const figure=node('figure'),frame=node('div',undefined,'image-frame'),img=node('img');img.src='api/image?'+new URLSearchParams({path:image.path});img.alt=`${image.camera} current RGB`;img.width=image.size[0];img.height=image.size[1];frame.append(img);figure.append(frame);
 figure.append(node('figcaption',`${image.camera} · ${image.size.join(' × ')} · ${view.image_source==='model_input'?'RGB actually sent to the model':'Raw observed RGB (model image input not recorded this round)'} · ${view.reference==='gripper_center'?'gripper center':'wrist'} reference point`));panel.append(figure);
 if(!overlay)return;
 const legend=node('div',undefined,'legend');for(const kind of ['current','target','issued','candidate']){const l=node('label',undefined,kind),check=node('input');check.type='checkbox';check.checked=state.layers.has(kind);check.onchange=()=>{if(check.checked)state.layers.add(kind);else state.layers.delete(kind);render();};l.append(check,node('span',kindNames[kind]));legend.append(l);}panel.append(legend);
 const info=node('div','Click a marker to see its 3D position, pixel position and source.','point-detail');panel.append(info);
 const missing=[];let shown=0;
 for(const marker of image.markers){if(!state.layers.has(marker.kind)||(state.arm!=='all'&&marker.arm!==state.arm))continue;
  const name=`${marker.arm==='left'?'Left':'Right'} · ${kindNames[marker.kind]}${marker.waypoint?' #'+marker.waypoint:''}`;
  if(!marker.pixel){missing.push(name+'：'+marker.reason);continue;}
  shown++;const b=node('button',`${marker.arm==='left'?'L':'R'}${marker.kind==='current'?'○':marker.kind==='issued'?'■':marker.kind==='candidate'?'◇':'+'}${marker.waypoint||''}`,'marker '+marker.kind+(marker.reference_only?' reference':''));b.setAttribute('aria-label',name);b.style.left=marker.percent[0]+'%';b.style.top=marker.percent[1]+'%';
  b.onclick=()=>{info.textContent=`${name}\n${marker.frame} · ${view.reference} · ${marker.pose.slice(0,3).map(n=>Number(n).toFixed(4)).join(', ')} m\nPixel u=${marker.pixel[0].toFixed(1)}, v=${marker.pixel[1].toFixed(1)}\nSource: ${marker.source}${marker.reference_only?'\nReference projection: calibration not verified':''}${marker.kind==='issued'?'\nIssued command; does not mean it was reached':''}`;};frame.append(b);
 }
 if(!shown)panel.append(node('p','No target projection to show. Projecting a known 3D target needs a valid calibration for the same frame.','notice'));
 if(image.markers.some(m=>m.reference_only&&m.pixel))panel.append(node('p','Dashed lines are reference projections: calibration not yet verified.','hint'));
 if(missing.length){const d=node('details');d.append(node('summary',`${missing.length} point(s) cannot be projected · view reason`),node('pre',missing.join('\n')));panel.append(d);}
 panel.append(node('p','Target projections are drawn on the current frame; issued does not mean reached.','hint'));
}
let busy=false;
async function refresh(){if(busy)return;busy=true;const version=selectionVersion;try{
 const query={offset:String(state.offset)};if(!state.follow&&state.episode)query.episode=state.episode;
 const data=await api('catalog',query);if(version!==selectionVersion)return;updateCatalog(data);remember();
 if(state.request){const requestId=state.request,detail=await api('request',{id:requestId});if(state.request!==requestId||version!==selectionVersion)return;
  const signature=requestId+':'+detail.request.revision+':'+detail.warnings.length;
  if(signature!==state.signature||detail.warnings.length){const changed=state.detail&&state.detail.request.id!==requestId;state.signature=signature;state.detail=detail;
   if(changed||state.follow||state.view<0)state.view=Math.max(0,detail.views.length-1);else state.view=Math.min(state.view,Math.max(0,detail.views.length-1));render();}
 }else{$('empty').hidden=false;$('detail').hidden=true;}
 }catch(error){$('connection').textContent='○ Disconnected · reconnecting';}finally{busy=false;if(version!==selectionVersion)refresh();}}
$('episodes').onchange=()=>{selectionVersion++;state.follow=false;state.episode=$('episodes').value;state.request=null;state.offset=0;state.view=-1;state.signature=null;refresh();};
$('live').onclick=()=>{selectionVersion++;state.follow=true;state.offset=0;state.signature=null;refresh();};
$('newer').onclick=()=>{selectionVersion++;state.follow=false;state.request=null;state.offset=Math.max(0,state.offset-100);refresh();};
$('older').onclick=()=>{selectionVersion++;state.follow=false;state.request=null;state.offset+=100;refresh();};
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>gotoTab(b.dataset.tab));
refresh();setInterval(refresh,1000);
