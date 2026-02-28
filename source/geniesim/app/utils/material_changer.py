# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, sys
import numpy as np
import omni.usd
import omni.kit.commands
from pxr import Sdf, UsdShade, Gf, UsdLux

from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.prims import get_prim_at_path


from pxr import Sdf
from pathlib import Path

from geniesim.plugins.logger import Logger
import geniesim.utils.system_utils as system_utils

logger = Logger()  # Create singleton instance


class material_changer:
    def __init__(self):
        pass

    def select_material(self, folder_name):
        selected_folder = str(system_utils.assets_path()) + "/" + folder_name
        base_color = None
        orm = None
        normal_map = None
        logger.info(selected_folder)
        for file in os.listdir(selected_folder):
            logger.info(file)
            if file.endswith("BaseColor.png"):
                base_color = os.path.join(selected_folder, file)
                logger.info(base_color)
            elif file.endswith("ORM.png"):
                orm = os.path.join(selected_folder, file)
            elif file.endswith(("NormalMap.png", "N.png", "Normal.png")):
                normal_map = os.path.join(selected_folder, file)
        return base_color, orm, normal_map, selected_folder

    def get_random_material_textures(self, base_folder):
        subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
        if not subfolders:
            raise FileNotFoundError("No subfolders found!")
        selected_folder = subfolders[np.random.randint(len(subfolders))]
        base_color = None
        orm = None
        normal_map = None

        for file in os.listdir(selected_folder):
            if file.endswith("BaseColor.png"):
                base_color = os.path.join(selected_folder, file)
            elif file.endswith("ORM.png"):
                orm = os.path.join(selected_folder, file)
            elif file.endswith(("NormalMap.png", "N.png", "Normal.png")):
                normal_map = os.path.join(selected_folder, file)
        if not (base_color and orm and normal_map):
            raise FileNotFoundError(
                "Some texture files are missing: please ensure BaseColor.png, ORM.png, and NormalMap.png exist in  "
                + selected_folder
            )

        return base_color, orm, normal_map

    def create_material(
        self,
        material_name,
        material_path,
        mdl_path,
        base_color_texture,
        orm_texture,
        normalmap_texture,
        texture_transform,
        metallic_amount,
        metallic_map_influence,
        roughness_map_influence,
    ):
        stage = omni.usd.get_context().get_stage()
        prim = get_prim_at_path(material_path)
        if not prim.IsValid():

            # Create a material
            omni.kit.commands.execute(
                "CreateMdlMaterialPrim",
                mtl_url=mdl_path,
                mtl_name="OmniPBR",
                mtl_path=material_path,
            )
        material_prim = stage.GetPrimAtPath(material_path)
        shader = UsdShade.Shader(omni.usd.get_shader_from_material(material_prim, get_prim=True))
        # Set material properties
        if base_color_texture:
            shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(base_color_texture)

        # ORMTexture-related settings
        if orm_texture:
            shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset).Set(orm_texture)
            shader.CreateInput("enable_ORM_texture", Sdf.ValueTypeNames.Bool).Set(True)
        # Normal map settings
        if normalmap_texture:
            shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(normalmap_texture)
        # Texture transformation and zoom settings
        shader.CreateInput("texture_translate", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(texture_transform["translate"]))
        shader.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float).Set(texture_transform["rotate"])
        shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(texture_transform["scale"]))
        # Metallicity setting
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic_amount)  # Set metallicity
        shader.CreateInput("metallic_texture_influence", Sdf.ValueTypeNames.Float).Set(
            metallic_map_influence
        )  # Set Metal Map Effect
        # Roughness affects settings
        shader.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(
            roughness_map_influence
        )  # Setting the roughness map effect
        # Enable UVW coordinate projection
        shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        # Enable world space
        shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)

        material = UsdShade.Material(material_prim)  # Make sure to use the UsdShade.Material object

        return material

    def assign_material(self, path, name):
        # Randomly select the texture map
        base_color_texture, orm_texture, normalmap_texture, texture_path = self.select_material(path)
        # Calling functions to create material examples
        material = self.create_material(
            material_name=name,
            material_path="/World/Materials/" + name,
            mdl_path="OmniPBR.mdl",
            base_color_texture=base_color_texture,
            orm_texture=orm_texture,
            normalmap_texture=normalmap_texture,
            texture_transform={
                "translate": (0.0, 0.0),
                "rotate": 0.0,
                "scale": (1.0, 1.0),
            },
            metallic_amount=0.5,  # Set metallicity
            metallic_map_influence=1,  # Set Metal Map Effect
            roughness_map_influence=0.5,  # Setting the roughness map effect
        )
        return material


class Light:
    def __init__(self, prim_path, stage, light_type, intensity, color, position, orientation, texture_file):
        self.prim_path = prim_path
        self.light_type = light_type
        self.stage = stage
        self.intensity = intensity
        self.color = color
        self.position = position
        self.orientation = orientation
        base_folder = str(system_utils.assets_path()) + "/" + texture_file
        for file in os.listdir(base_folder):
            if file.endswith(".hdr"):
                self.texture_file = os.path.join(base_folder, file)

    def initialize(self):
        if self.light_type == "Dome":
            light = UsdLux.DomeLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)
            light.CreateTextureFileAttr().Set(Sdf.AssetPath(self.texture_file))
        elif self.light_type == "Sphere":
            light = UsdLux.SphereLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)
        elif self.light_type == "Disk":
            light = UsdLux.DiskLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)
        elif self.light_type == "Rect":
            light = UsdLux.RectLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)
        elif self.light_type == "Distant":
            light = UsdLux.DistantLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)

        light.CreateEnableColorTemperatureAttr().Set(True)
        lightPrim = SingleXFormPrim(self.prim_path, position=self.position, orientation=self.orientation)

        return lightPrim
