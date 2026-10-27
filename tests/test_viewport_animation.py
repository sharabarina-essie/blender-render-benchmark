from types import SimpleNamespace as Item
import unittest
from viewport_animation import action_has_fcurves


class ViewportAnimationTests(unittest.TestCase):
    def test_legacy_actions_and_missing_animation(self):
        self.assertFalse(action_has_fcurves(None))
        self.assertFalse(action_has_fcurves(Item(action=None)))
        for curves in ([], [object()]):
            self.assertEqual(action_has_fcurves(Item(action=Item(fcurves=curves))),bool(curves))

    def test_layered_actions_only_count_the_assigned_slot(self):
        selected, other=object(), object()
        strip=Item(type='KEYFRAME',channelbag=lambda slot: Item(fcurves=[object()]) if slot is selected else None)
        action=Item(layers=[Item(strips=[strip])])
        self.assertTrue(action_has_fcurves(Item(action=action,action_slot=selected)))
        self.assertFalse(action_has_fcurves(Item(action=action,action_slot=other)))
        self.assertFalse(action_has_fcurves(Item(action=action,action_slot=None)))

    def test_nonkeyframe_strips_do_not_get_queried(self):
        def forbidden(_):
            self.fail('Non-keyframe strip was queried')
        action=Item(layers=[Item(strips=[Item(type='OTHER',channelbag=forbidden)])])
        self.assertFalse(action_has_fcurves(Item(action=action,action_slot=object())))
