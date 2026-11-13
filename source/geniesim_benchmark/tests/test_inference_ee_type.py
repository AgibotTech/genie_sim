"""Exercise the wire payload without loading Isaac or connecting to a policy server."""
import ast
from pathlib import Path
from types import SimpleNamespace
import time
import unittest

ROOT = Path(__file__).parents[1] / 'src/geniesim_benchmark'


def load_contract():
    # Config functions are unused here; retain the real config declarations.
    tree = ast.parse((ROOT / 'utils/name_utils.py').read_text())
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
    ns = {alias.name: lambda *args: None for n in imports for alias in n.names}
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, 'name_utils.py', 'exec'), ns)
    policy = ast.parse((ROOT / 'benchmark/policy/corobotpolicy.py').read_text())
    cls = next(n for n in policy.body if isinstance(n, ast.ClassDef) and n.name == 'CoRobotPolicy')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_payload')
    scope = {'time': time}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'corobotpolicy.py', 'exec'), scope)
    return ns['ROBOT_CONFIGS'], scope['get_payload']


class EndEffectorWireContract(unittest.TestCase):
    def test_wire_payload_uses_canonical_robot_and_nonempty_ee(self):
        configs, get_payload = load_contract()
        for robot, ee in [('G2_crsB_90d_wbc', 'zhiyuan_gripper_ctek90d'),
                          ('G2_crsB_90d', 'zhiyuan_gripper_ctek90d'),
                          ('G2_crsB_wuji', 'wuji'),
                          ('G1_omnipicker', 'zhiyuan_gripper_omnipicker'),
                          ('G2_omnipicker', 'zhiyuan_gripper_omnipicker'),
                          ('G2_crsB_omnipicker', 'zhiyuan_gripper_omnipicker'),
                          ('arx', 'arx_gripper'), ('agilex', 'agilex_gripper')]:
            with self.subTest(robot=robot):
                cfg = configs[robot]
                self.assertNotIn('eef_type', cfg)
                self.assertEqual(cfg['ee_type'], ee)
                fake = SimpleNamespace(
                    _pre_process_obs=lambda obs, _: obs,
                    _split_states=lambda _: ([0.] * 14, [0., 0.], [], [], [1., 2., 3.]),
                    _encode_image_jpeg=lambda _: b'image', _build_end_pose=lambda _: [],
                    _attach_depth_and_calibration=lambda *args: None,
                    _robot_type=cfg['robot_type'], _ee_type=cfg['ee_type'],
                    sub_task_name='test', _current_episode_idx=0, _episode_done=False,
                    _hist_frame_interval=0, _history_buffer=[], debug=False, preview=False,
                )
                out = get_payload(fake, {'states': {}, 'images': dict.fromkeys(['head', 'left_hand', 'right_hand'])}, 'test', None)['params']
                self.assertEqual(out['ee_type'], ee)
                expected_robot = {'G1_omnipicker': 'a2d_sim', 'G2_omnipicker': 'g2a_sim',
                                  'G2_crsB_omnipicker': 'g2a_sim', 'arx': 'arx_sim',
                                  'agilex': 'agilex_sim'}.get(robot, 'g2a_sim')
                self.assertEqual(out['robot_type'], expected_robot)
                self.assertEqual(out['states']['chassis_states'], [1., 2., 3.])


    def test_wuji_native_payloads_include_explicit_ee_type(self):
        import __future__
        for relative in ['libuipc/adapter.py', 'newton/corobot.py']:
            with self.subTest(path=relative):
                tree = ast.parse((ROOT / relative).read_text())
                method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_payload')
                groups = SimpleNamespace(head=(), left_arm=(), right_arm=(), waist=(), left_effector=(), right_effector=())
                scope = dict(WUJI_GROUPS=groups, time=time, _finite_positions=lambda data, _: data,
                             encode_image=lambda image: image, _history_interval=lambda x: x,
                             HEAD_JOINTS=(), WAIST_JOINTS=(), ARM_JOINTS=(), GRIPPER_JOINTS=(),
                             finite=lambda v: True, wire_images=lambda images: images,
                             end_pose=lambda *args, **kwargs: None)
                exec(compile(ast.Module(body=[method], type_ignores=[]), relative, 'exec',
                             flags=__future__.annotations.compiler_flag), scope)
                if relative.startswith('libuipc'):
                    result = scope['build_payload']({'snapshot': {'captured_wall_time_ns': 1, 'joints': {}}, 'images': {}},
                                                    {}, instruction='test', task_name='fold_cloth')
                else:
                    result = scope['build_payload']({}, dict.fromkeys(['head', 'left_hand', 'right_hand'], b'image'),
                                                    'test', task_name='wok_flip')
                self.assertEqual(result['params']['robot_type'], 'g2a_sim')
                self.assertEqual(result['params']['ee_type'], 'wuji')


if __name__ == '__main__':
    unittest.main()
