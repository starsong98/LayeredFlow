import bpy
from numpy.random import uniform as U, normal as N, randint as RI

def is_material_seethrough(material):
    return material.name.startswith('glass') or material.name.startswith('refraction')

def is_obj_has_transparent_material(obj):
    for i, material in enumerate(obj.data.materials):
        if is_material_seethrough(material):
            return True

def is_material_emission(material):
    if material.use_nodes:
        for node in material.node_tree.nodes:
            if node.type == 'EMISSION':
                return True
    return False

def is_obj_has_emission_material(obj):
    for i, material in enumerate(obj.data.materials):
        if material is not None and is_material_emission(material):
            return True
    return False
        
def emission_invisible(scene):
    for obj in scene.objects:
        if obj.type == 'MESH':
            materials = obj.data.materials
            if materials is None or len(materials) != 1:
                continue
            if is_obj_has_emission_material(obj):
                obj.visible_camera = False
                obj.visible_transmission = False

def assign_glass_material(obj, color=(1, 1, 1, 1), ior=1.45, translucency=0):
    glass_material = bpy.data.materials.new(name="glass_material")
    glass_material.use_nodes = True
    while glass_material.node_tree.nodes:
        glass_material.node_tree.nodes.remove(glass_material.node_tree.nodes[0])

    random_color = (1-U(0, 0.1), 1-U(0, 0.1), 1-U(0, 0.1), 1)

    nodes = glass_material.node_tree.nodes
    glass = nodes.new(type='ShaderNodeBsdfGlass')
    glass.inputs['Color'].default_value = random_color # white color glass
    glass.inputs['IOR'].default_value = ior # index of refraction for glass
    glass.inputs['Roughness'].default_value = U(0, 0.01)

    output = glass_material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    links = glass_material.node_tree.links
    link = links.new(glass.outputs[0], output.inputs[0])

    if obj.type in ['MESH', 'CURVE']:
        obj.data.materials.clear()
        obj.data.materials.append(glass_material)
    else:
        raise Exception(f"Object type is {obj.type}, not mesh")

def assign_rough_glass_material(obj, color=(1, 1, 1, 1), ior=1.45, translucency=0):
    glass_material = bpy.data.materials.new(name="rough_glass_material")
    glass_material.use_nodes = True
    while glass_material.node_tree.nodes:
        glass_material.node_tree.nodes.remove(glass_material.node_tree.nodes[0])

    random_color = (1-U(0, 0.1), 1-U(0, 0.1), 1-U(0, 0.1), 1)

    nodes = glass_material.node_tree.nodes
    glass = nodes.new(type='ShaderNodeBsdfGlass')
    glass.inputs['Color'].default_value = random_color # white color glass
    glass.inputs['IOR'].default_value = ior # index of refraction for glass
    glass.inputs['Roughness'].default_value = U(0.05, 0.3) # index of refraction for glass

    translucent = nodes.new(type='ShaderNodeBsdfTranslucent')
    translucent.inputs[0].default_value = color

    mix = nodes.new(type='ShaderNodeMixShader')
    mix.inputs[0].default_value = translucency

    output = glass_material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')

    links = glass_material.node_tree.links
    link = links.new(glass.outputs[0], mix.inputs[1])
    link = links.new(translucent.outputs[0], mix.inputs[2])
    # link = links.new(mix.outputs[0], output.inputs[0])
    link = links.new(glass.outputs[0], output.inputs[0])

    if obj.type in ['MESH', 'CURVE']:
        obj.data.materials.clear()
        obj.data.materials.append(glass_material)
    else:
        raise Exception(f"Object type is {obj.type}, not mesh")

def add_randomness_to_glass(obj):
    random_color = (1-U(0, 0.1), 1-U(0, 0.1), 1-U(0, 0.1), 1)

    for material in obj.data.materials:
        if material.name.startswith('glass'):
            nodes = material.node_tree.nodes
            for node in nodes:
                if node.type == 'BSDF_GLASS':
                    node.inputs['Color'].default_value = random_color # white color glass
                    node.inputs['IOR'].default_value = 1.45 # index of refraction for glass
                    node.inputs['Roughness'].default_value = U(0, 0.01) # index of refraction for glass

def assign_refraction_material(obj, color=(1, 1, 1, 1), ior=1.45, roughness=0.0):
    refraction_material = bpy.data.materials.new(name="refraction_material")  
    refraction_material.use_nodes = True
    while refraction_material.node_tree.nodes:
        refraction_material.node_tree.nodes.remove(refraction_material.node_tree.nodes[0])
    
    nodes = refraction_material.node_tree.nodes
    refraction = nodes.new(type='ShaderNodeBsdfRefraction')
    refraction.inputs['Color'].default_value = color # white color glass
    refraction.inputs['IOR'].default_value = ior # index of refraction for glass
    refraction.inputs['Roughness'].default_value = 0
    output_node = refraction_material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')

    refraction_material.node_tree.links.new(refraction.outputs['BSDF'], output_node.inputs['Surface'])
    if obj.type in ['MESH', 'CURVE']:
        obj.data.materials.clear()
        obj.data.materials.append(refraction_material)
    else:
        raise Exception(f"Object type is {obj.type}, not mesh")

def assign_metalic_material(obj):
    random_color = (1-U(0, 0.2), 1-U(0, 0.2), 1-U(0, 0.2), 1)
    metalic_material = bpy.data.materials.new(name="metalic_material")
    metalic_material.use_nodes = True
    while metalic_material.node_tree.nodes:
        metalic_material.node_tree.nodes.remove(metalic_material.node_tree.nodes[0])

    metalic_node = metalic_material.node_tree.nodes.new(type='ShaderNodeBsdfAnisotropic')
    metalic_node.inputs['Color'].default_value = random_color # white color metalic
    metalic_node.inputs['Roughness'].default_value = 0.3 + max(-0.1, N(0, 0.1)) # index of refraction for glass
    output_material_node = metalic_material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')

    metalic_material.node_tree.links.new(metalic_node.outputs['BSDF'], output_material_node.inputs['Surface'])
    if obj.type in ['MESH', 'CURVE']:
        if obj.data.materials:
            obj.data.materials[0] = metalic_material
        else:
            obj.data.materials.append(metalic_material)
    else:
        raise Exception(f"Object type is {obj.type}, not mesh")

def set_node_color(node, color):
    for input in node.inputs:
        if 'Color' in input.name:
            input.default_value = color


# iterate over all materials
def set_all_color(color):
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                set_node_color(node, color)

def set_roughness(material, value):
    # Iterating through the nodes in the material
    for node in material.node_tree.nodes:
        # Checking if the node is of type 'ShaderNodeBsdfPrincipled'
        if 'Roughness' in node.inputs:
            # Setting the roughness value
            roughness = node.inputs['Roughness'].default_value
            if roughness < value:
                node.inputs['Roughness'].default_value = value


def set_color_to_all(scene):
    # Iterate over all materials in the data
    for material in bpy.data.materials:
        # Check if the material name is not 'glass'
        material.use_nodes = True
        nodes = material.node_tree.nodes
        if not material.name.startswith('glass'):
            for node in nodes:
                if node.type != 'OUTPUT_MATERIAL':
                    nodes.remove(node)
            diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
            diffuse.inputs[0].default_value = (1, 1, 1, 1)

            material_output = None
            for node in nodes:
                if node.type == 'OUTPUT_MATERIAL':
                    material_output = node
            if material_output is None:
                material_output = nodes.new(type='ShaderNodeOutputMaterial')

            links = material.node_tree.links
            link = links.new(diffuse.outputs[0], material_output.inputs[0])
        else:
            color = (1, 1, 1, 1)
            ior = 1.45
            roughness = 0
            for node in nodes:
                if node.type == 'BSDF_GLASS':
                    color = node.inputs['Color'].default_value
                    ior = node.inputs['IOR'].default_value
                    roughness = node.inputs['Roughness'].default_value
            
            refraction = nodes.new(type='ShaderNodeBsdfRefraction')
            refraction.inputs['Color'].default_value = color # white color glass
            refraction.inputs['IOR'].default_value = ior # index of refraction for glass
            refraction.inputs['Roughness'].default_value = roughness

            material_output = None
            for node in nodes:
                if node.type == 'OUTPUT_MATERIAL':
                    material_output = node
            if material_output is None:
                material_output = nodes.new(type='ShaderNodeOutputMaterial')
                
            links = material.node_tree.links
            link = links.new(refraction.outputs[0], material_output.inputs[0])

def set_light_to_white(scene):
    white_color = (1, 1, 1)
    for obj in scene.objects:
        if obj.type == 'LIGHT':
            obj.data.color = white_color
            obj.data.energy = 5000

def create_spot_material(obj, point, size=0.1):
    mat = bpy.data.materials.new(name="Spot_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Add shader nodes
    pink_rgb = nodes.new(type="ShaderNodeRGB")
    pink_rgb.outputs[0].default_value = (1, 0, 1, 1)
    white_rgb = nodes.new(type="ShaderNodeRGB")
    white_rgb.outputs[0].default_value = (1, 1, 1, 1)
    pink_bsdf = nodes.new(type="ShaderNodeBsdfDiffuse")
    white_bsdf = nodes.new(type="ShaderNodeBsdfDiffuse")
    mat.node_tree.links.new(pink_rgb.outputs["Color"], pink_bsdf.inputs["Color"])
    mat.node_tree.links.new(white_rgb.outputs["Color"], white_bsdf.inputs["Color"])

    shader_mix = nodes.new(type="ShaderNodeMixShader")
    shader_output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(white_bsdf.outputs["BSDF"], shader_mix.inputs[1])
    mat.node_tree.links.new(pink_bsdf.outputs["BSDF"], shader_mix.inputs[2])

    spot_node = nodes.new(type="ShaderNodeVectorMath")
    spot_node.inputs[0].default_value = point.to_tuple()
    spot_node.operation = 'DISTANCE'

    geometry_node = nodes.new(type="ShaderNodeNewGeometry")
    mat.node_tree.links.new(geometry_node.outputs["Position"], spot_node.inputs[1])

    # Set spot size and mix shaders
    math_node = nodes.new(type="ShaderNodeMath")
    math_node.operation = 'LESS_THAN'
    math_node.inputs[1].default_value = size
    mat.node_tree.links.new(spot_node.outputs['Value'], math_node.inputs[0])

    mat.node_tree.links.new(math_node.outputs[0], shader_mix.inputs[0])
    mat.node_tree.links.new(shader_mix.outputs["Shader"], shader_output.inputs["Surface"])

    # Assign material to object
    obj.data.materials.clear()
    obj.data.materials.append(mat)
