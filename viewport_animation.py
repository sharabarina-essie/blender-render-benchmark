"""Animation presence across legacy and slot/layer Blender Actions."""


def action_has_fcurves(animation_data):
    if animation_data is None or animation_data.action is None:
        return False
    action = animation_data.action
    curves = getattr(action, 'fcurves', None)
    if curves is not None:
        return bool(curves)
    slot = getattr(animation_data, 'action_slot', None)
    if slot is None:
        return False
    # Matches Blender 5.2 bpy_extras.anim_utils.action_get_channelbag_for_slot:
    # only curves belonging to this object's assigned slot count as animation.
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type != 'KEYFRAME':
                continue
            bag = strip.channelbag(slot)
            if bag is not None and bag.fcurves:
                return True
    return False
