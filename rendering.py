import torch
import numpy as np
import imageio
import pytorch3d
from pytorch3d.ops import cubify
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    HardPhongShader,
    PointLights,
    FoVPerspectiveCameras,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds, Meshes

def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        device = get_device()
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        device = get_device()
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def render_mesh(
        meshes: Meshes,
        path: str = None,
        image_size=512,
        retexture=True,
        color=[0, 0, 1],
        light_location=[[0, 0, -3]],
        dist=3,
        elev=0,
        azim_start=-180,
        azim_end=180,
        num_views=30,
        fps=15,
        device=None
):
    if device is None:
        device = get_device()
    
    renderer = get_mesh_renderer(image_size, device=device)

    if retexture:
        textures = torch.ones_like(meshes.verts_list()[0]) * torch.tensor(color, device=device)
        meshes.textures = TexturesVertex([textures])

    R, T = look_at_view_transform(
        dist=dist,
        elev=elev,
        azim=azim_start if num_views<=1 else np.linspace(azim_start, azim_end, num_views, endpoint=False),
        device=device
    )

    # Prepare the camera:
    cameras = FoVPerspectiveCameras(
        R=R, T=T, fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = PointLights(location=light_location, device=device)

    image = renderer(meshes.extend(num_views), cameras=cameras, lights=lights)
    image = image.cpu().detach().numpy()[..., :3]
    image = (image*255).astype(np.uint8)

    if path is not None:
        imageio.mimsave(path, image, fps=fps, loop=0)
    return image

def render_pointcloud(
        points: torch.Tensor,
        path: str = None,
        image_size=512,
        color=[0, 0, 1],
        light_location=[[0, 0, -3]],
        dist=3,
        elev=0,
        azim_start=-180,
        azim_end=180,
        num_views=30,
        fps=15,
        device=None
):
    if device is None:
        device = get_device()
    
    renderer = get_points_renderer(image_size, device=device)

    rgb = torch.ones_like(points) * torch.tensor(color, device=device)
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=2)

    pointclouds = Pointclouds(points=points, features=rgb)


    R, T = look_at_view_transform(
        dist=dist,
        elev=elev,
        azim=azim_start if num_views<=1 else np.linspace(azim_start, azim_end, num_views, endpoint=False),
        device=device
    )

    # Prepare the camera:
    cameras = FoVPerspectiveCameras(
        R=R, T=T, fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = PointLights(location=light_location, device=device)

    image = renderer(pointclouds.extend(num_views), cameras=cameras, lights=lights)
    image = image.cpu().detach().numpy()[..., :3]
    image = (image*255).astype(np.uint8)

    if path is not None:
        imageio.mimsave(path, image, fps=fps, loop=0)
    return image

def render_voxels(
        voxels: torch.Tensor,
        path: str = None,
        thresh=0.5,
        image_size=512,
        color=[0, 0, 1],
        light_location=[[0, 0, -3]],
        dist=3,
        elev=0,
        azim_start=-180,
        azim_end=180,
        num_views=30,
        fps=15,
        device=None
):
    if len(voxels.shape) > 4:
        voxels = voxels.squeeze(0)

    meshes = cubify(voxels, thresh, device)

    return render_mesh(
        meshes,
        path,
        image_size,
        True,
        color,
        light_location,
        dist,
        elev,
        azim_start,
        azim_end,
        num_views,
        fps,
        device
    )

def cubify_voxels_confidence(
        voxels: torch.Tensor,
        max_conf=0.5,
        min_conf=0.1,
        device = None):
    
    if device is None:
        device = get_device()
    
    D, H, W = voxels.size()

    cube_verts = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.float32,
        device=device)
    

    cube_faces = torch.tensor(
        [
            [0, 1, 2],
            [1, 3, 2],  # left face: 0, 1
            [2, 3, 6],
            [3, 7, 6],  # bottom face: 2, 3
            [0, 2, 6],
            [0, 6, 4],  # front face: 4, 5
            [0, 5, 1],
            [0, 4, 5],  # up face: 6, 7
            [6, 7, 5],
            [6, 5, 4],  # right face: 8, 9
            [1, 7, 3],
            [1, 5, 7],  # back face: 10, 11
        ],
        dtype=torch.int64,
        device=device,
    )

    voxel_cube_faces = []
    voxel_cube_verts = []
    voxel_cube_textures = []

    low_conf_color = torch.tensor([0.0, 1.0, 0.0], device=device)
    high_conf_color = torch.tensor([1.0, 0.0, 0.0], device=device)

    i = 0
    for d in range(D):
        for h in range(H):
            for w in range(W):
                if voxels[d, h, w] < min_conf:
                    continue
                conf = torch.round(voxels[d, h, w], decimals=1)

                conf_rescaled = torch.clip(conf, 0.0, max_conf) / max_conf

                voxel_cube_verts.append(cube_verts * conf_rescaled + torch.tensor([d, h, w], device=device))

                voxel_cube_faces.append(cube_faces + 8*i)

                voxel_cube_textures.append((conf * high_conf_color + (1-conf) * low_conf_color).repeat(8, 1))

                i += 1
    
    voxel_cube_faces = torch.cat(voxel_cube_faces, dim=0)
    voxel_cube_verts = torch.cat(voxel_cube_verts, dim=0)
    voxel_cube_verts = voxel_cube_verts / torch.tensor([D, H, W], device=device) - torch.tensor([0.5, 0.5, 0.5], device=device)
    voxel_cube_textures = torch.cat(voxel_cube_textures, dim=0)

    meshes = Meshes(
        verts=[voxel_cube_verts],
        faces=[voxel_cube_faces],
        textures=TexturesVertex([voxel_cube_textures])
    )

    return meshes

