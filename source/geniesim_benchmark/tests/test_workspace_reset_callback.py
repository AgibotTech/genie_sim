"""World.play callbacks must not sample handles invalidated by cross-scene reset."""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

SOURCE = Path(__file__).parents[1]/'src/geniesim_benchmark/app/controllers/api_core.py'

def methods():
    tree=ast.parse(SOURCE.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='_reinitialize_workspace' for m in n.body))
    selected=[m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name in ['_reinitialize_workspace','on_ros_tick']]
    scope=dict(logger=SimpleNamespace(info=lambda *args:None,warning=lambda *args:None,error=lambda *args:None),
               time=SimpleNamespace(sleep=lambda _:None),update_stage=lambda:None)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec'),scope)
    return scope

class WorkspaceCallbackTests(unittest.TestCase):
    def test_callbacks_resume_only_after_new_handles_registered(self):
        functions=methods();samples=[];state={'ready':True};api=SimpleNamespace()
        def tick(*args):
            self.assertTrue(state['ready']);samples.append(args)
        def stop():state['ready']=False
        def play():api.on_ros_tick(.01)
        world=SimpleNamespace(is_playing=lambda:True,stop=stop,play=play,current_time=1,current_time_step_index=2)
        robot=SimpleNamespace(tick=tick,register_joint_state=lambda articulation:state.update(ready=True))
        api.__dict__.update(ui_builder=SimpleNamespace(my_world=world,initialize_articulation=lambda:None),robot_interface=robot,
                            _robot_base_anchored=lambda:True,_get_articulation=lambda:object(),
                            sync_drive_targets_to_current=lambda *args,**kw:None,local_recorder=SimpleNamespace(is_recording=False),enable_ros=False)
        api.on_ros_tick=MethodType(functions['on_ros_tick'],api)
        functions['_reinitialize_workspace'](api)
        self.assertEqual(samples,[])
        self.assertFalse(api._workspace_reinitializing)
        api.on_ros_tick(.01)
        self.assertEqual(len(samples),1)
        def fail():raise RuntimeError('new view failed')
        api.ui_builder.initialize_articulation=fail
        with self.assertRaises(RuntimeError):functions['_reinitialize_workspace'](api)
        api.on_ros_tick(.01)
        self.assertTrue(api._workspace_reinitializing)
        self.assertEqual(len(samples),1)

if __name__=='__main__':unittest.main()
