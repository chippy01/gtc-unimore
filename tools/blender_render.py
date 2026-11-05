"""
Blender side of tools/render_figures.py: render a scene once per texture, swapping only the image it names.
  blender -b scene.blend --python tools/blender_render.py -- jobs.json
jobs.json: {"image": name of the image datablock to swap,
            "jobs": [{"texture": path, "output": path, "percent": 100,
                      "border": [x0, y0, x1, y1] (optional, fractions of the frame, top-left origin)}, ...]}
Every other setting comes from the .blend file, which is never saved.
"""
import json
import sys

import bpy


def image_nodes(image_name):
    """Every Image Texture node that shows image_name, with its material."""
    found = []
    for material in bpy.data.materials:
        if material.use_nodes and material.users:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None and node.image.name == image_name:
                    found.append((material, node))
    return found


def main():
    with open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8") as file:
        spec = json.load(file)
    nodes = image_nodes(spec["image"])
    if not nodes:
        names = sorted(image.name for image in bpy.data.images)
        raise RuntimeError(f"No Image Texture node shows '{spec['image']}', the images are: {names}")
    colorspace = nodes[0][1].image.colorspace_settings.name
    render = bpy.context.scene.render
    for job in spec["jobs"]:
        image = bpy.data.images.load(job["texture"])
        image.colorspace_settings.name = colorspace
        for material, node in nodes:
            node.image = image
            material.node_tree.nodes.active = node  # Workbench's Texture color shows the active image node
        render.resolution_percentage = job.get("percent", 100)
        border = job.get("border")
        render.use_border = render.use_crop_to_border = border is not None
        if border is not None:
            x0, y0, x1, y1 = border
            render.border_min_x, render.border_max_x = x0, x1
            render.border_min_y, render.border_max_y = 1.0 - y1, 1.0 - y0  # Blender counts from the bottom
        render.filepath = job["output"]
        bpy.ops.render.render(write_still=True)
        print(f"Rendered '{job['output']}'")


main()
