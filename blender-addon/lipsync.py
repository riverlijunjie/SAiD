from math import pi
import os
from typing import Optional, Set
import aud
import bpy


bl_info = {
    "name": "Lipsync",
    "author": "Inkyu",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Lipsync",
    "description": "Tools for generating lipsync animation",
    "category": "Lipsync",
}


class LipsyncProperty(bpy.types.PropertyGroup):
    """Store the properties"""

    audio_path: bpy.props.StringProperty(
        name="Audio Path",
        default="",
        description="Path of the audio file",
        subtype="FILE_PATH",
    )

    mesh_sequence_dir: bpy.props.StringProperty(
        name="Mesh Sequence Dir",
        default="",
        description="Directory of the mesh sequence",
        subtype="DIR_PATH",
    )


class Lipsync_PT_MainPanel(bpy.types.Panel):
    """Main control panel"""

    bl_idname = "LIPSYNC_PT_main_panel"
    bl_label = "Main Panel for lipsync"
    bl_category = "Lipsync"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context: bpy.types.Context):
        """Draw UI elements

        Parameters
        ----------
        context : bpy.types.Context
            Blender context
        """
        box = self.layout.box()

        # Input data for the lipsync animation
        row = box.row()
        row.prop(context.scene.lipsync_property, "audio_path")

        row = box.row()
        row.prop(context.scene.lipsync_property, "mesh_sequence_dir")

        # Button
        row = box.row()
        row.operator("lipsync.generate_operator", text="Generate")


class LipsyncGenerateOperator(bpy.types.Operator):
    """Operator for the 'Generate' button"""

    bl_idname = "lipsync.generate_operator"
    bl_label = "lipsync.generate_operator"

    def execute(self, context: bpy.types.Context) -> Set[str]:
        """Execute the operator

        Parameters
        ----------
        context : bpy.types.Context
            Blender context

        Returns
        -------
        Set[str]
            Set of status messages: https://docs.blender.org/api/current/bpy.types.Operator.html
        """
        audio_path = context.scene.lipsync_property.audio_path
        mesh_sequence_dir = context.scene.lipsync_property.mesh_sequence_dir

        # Reset the scene frame to 1
        context.scene.frame_set(1)

        # Create collection
        collection = bpy.data.collections.new("Lipsync")
        context.scene.collection.children.link(collection)

        # Load the sound
        bpy.ops.object.speaker_add(rotation=(pi, 0.0, 0.0))
        speaker = context.object
        speaker.hide_set(True)
        speaker.data.sound = bpy.data.sounds.load(audio_path)
        speaker.data.attenuation = 0.0
        speaker.data.update_tag()
        context.collection.objects.unlink(speaker)
        collection.objects.link(speaker)

        # List the mesh sequence
        sequence = []
        for path in sorted(os.listdir(mesh_sequence_dir)):
            abs_path = os.path.join(mesh_sequence_dir, path)
            if os.path.isfile(abs_path):
                sequence.append(abs_path)

        # Update frame rate of the animation
        num_sequence = len(sequence)
        samplerate = speaker.data.sound.samplerate

        sd = aud.Sound(audio_path)
        length = sd.length
        del sd

        context.scene.render.fps = num_sequence
        context.scene.render.fps_base = length / samplerate

        # Generate animation sequence
        obj = self.load_obj(context, sequence[0])
        if obj is None:
            return {"CANCELLED"}

        context.collection.objects.unlink(obj)
        collection.objects.link(obj)
        obj.name = "Object"
        obj.data.name = "Object"

        num_vertices = len(obj.data.vertices)

        for sdx in range(1, num_sequence):
            obj_tmp = self.load_obj(context, sequence[sdx])
            if obj_tmp is None:
                return {"CANCELLED"}

            for vdx in range(num_vertices):
                obj.data.vertices[vdx].co = obj_tmp.data.vertices[vdx].co
                obj.data.vertices[vdx].keyframe_insert("co", frame=sdx + 1)

            bpy.ops.object.delete()

        return {"FINISHED"}

    def load_obj(
        self, context: bpy.types.Context, path: str
    ) -> Optional[bpy.types.Object]:
        ext = os.path.splitext(path)[-1]

        obj = None

        if ext == ".ply":
            bpy.ops.import_mesh.ply(filepath=path)
            obj = context.object
        elif ext == ".obj":
            bpy.ops.import_scene.obj(filepath=path)
            obj = context.selected_objects[0]
        else:
            self.report({"ERROR_INVALID_INPUT"}, f"{ext} is not supported")

        return obj


# List of classes to be registered
classes = (
    LipsyncProperty,
    Lipsync_PT_MainPanel,
    LipsyncGenerateOperator,
)


def register():
    """Register the classes"""
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.lipsync_property = bpy.props.PointerProperty(type=LipsyncProperty)


def unregister():
    """Unregister the classes"""
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.lipsync_property


if __name__ == "__main__":
    register()
