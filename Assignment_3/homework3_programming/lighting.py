"""
- Lambertian diffuse sphere
- Distance attenuation (point light)
- Blinn-Phong specular
Assume the intensity of the light is 1

Outputs: 3 PNG images + displays them
"""

import numpy as np
import matplotlib.pyplot as plt


def normalize(v, axis=None, eps=1e-12):
    """Safely normalize vector(s). If axis is None: normalize 1D vector.
    If axis is given: normalize along that axis for arrays (e.g., axis=2 for HxWx3)."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def render_shaded_sphere(size=400, radius=1.0):
    """Create sphere surface points and normals in camera space.
    Returns:
      P: (H,W,3) positions on sphere (z>=0), undefined outside mask
      N: (H,W,3) unit normals, undefined outside mask
      mask: (H,W) boolean inside sphere
      xx, yy, zz: (H,W) coordinate grids for convenience
    """
    x = np.linspace(-radius, radius, size)
    y = np.linspace(-radius, radius, size)
    xx, yy = np.meshgrid(x, y)

    rr2 = xx**2 + yy**2
    mask = rr2 <= radius**2

    zz = np.zeros_like(xx)
    zz[mask] = np.sqrt(np.maximum(radius**2 - rr2[mask], 0.0))

    P = np.stack([xx, yy, zz], axis=2) # 3D surface points
    N = normalize(P, axis=2) # Unit normals for shading

    return P, N, mask, xx, yy, zz


def lambertian_diffuse(N, L, albedo=1.0):
    """Lambertian diffuse.
    
    Args:
        N: (H,W,3) unit surface normals
        L: (H,W,3) unit light direction vectors
        albedo: (float) surface reflectance
    
    Returns:
        I: (H,W) diffuse intensity
    """
    # Lambertian diffuse: I = albedo * max(0, N · L)
    cos_theta = np.sum(N * L, axis=2)  # Element-wise dot product along last axis
    I = albedo * np.maximum(cos_theta, 0.0)
    return I 


def point_light_attenuated_diffuse(P, N, light_pos, albedo=1.0):
    """Diffuse with point light and inverse-square distance attenuation.
    
    Args:
        P: (H,W,3) surface positions
        N: (H,W,3) unit surface normals
        light_pos: (3,) light source position
        albedo: (float) surface reflectance
    
    Returns:
        I: (H,W) diffuse intensity with distance attenuation
    """
    Lvec = light_pos.reshape(1, 1, 3) - P  # from surface to light
    dist = np.linalg.norm(Lvec, axis=2)
    L = normalize(Lvec, axis=2)  # direction of incident light
    
    # Diffuse with inverse-square distance attenuation
    # I = albedo * max(0, N · L) / dist^2
    cos_theta = np.sum(N * L, axis=2)  # dot product
    attenuation = 1.0 / (dist ** 2 + 1e-8)  # avoid division by zero
    I = albedo * np.maximum(cos_theta, 0.0) * attenuation
    return I


def blinn_phong_specular(P, N, light_pos, V, s=0.5, x=50):
    """Blinn-Phong specular term with distance attenuation.
    
    Args:
        P: (H,W,3) surface positions
        N: (H,W,3) unit surface normals
        light_pos: (3,) light source position
        V: (3,) view direction (from surface towards camera)
        s: (float) specular reflectance
        x: (float) material shininess exponent
    
    Returns:
        I: (H,W) specular intensity
    """
    Lvec = light_pos.reshape(1, 1, 3) - P  # from surface to light
    dist = np.linalg.norm(Lvec, axis=2)
    L = normalize(Lvec, axis=2)  # direction of incident light
    
    # Blinn-Phong: H = (L + V) / |L + V|
    V_expanded = V.reshape(1, 1, 3) * np.ones_like(N)  # expand view direction to match shape
    H = L + V_expanded  # halfway vector
    H = normalize(H, axis=2)  # normalize
    
    # Specular intensity: I_spec = s * max(0, N · H)^x with distance attenuation
    cos_theta = np.sum(N * H, axis=2)  # dot product
    attenuation = 1.0 / (dist ** 2 + 1e-8)  # inverse-square distance attenuation
    I = s * (np.maximum(cos_theta, 0.0) ** x) * attenuation
    return I 


def show_and_save(img, title, filename, cmap=None, vmin=None, vmax=None):
    
    plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
    else:
        plt.imshow(np.clip(img, 0.0, 1.0))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


def main():
    size = 400
    radius = 1.0

    # Build sphere geometry
    P, N, mask, *_ = render_shaded_sphere(size=size, radius=radius)

    # Common setup
    albedo = 1.0
    bg = 0.0 # background

    # -------------------------
    # 1) Diffuse only (directional light, idealized point light)
    # -------------------------
    light_dir = normalize(np.array([1.0, 1.0, 1.0]))  # direction of incident light
    
    L_dir = light_dir.reshape(1, 1, 3) * np.ones_like(N)


    I_diffuse = lambertian_diffuse(N, L_dir, albedo=albedo)
    I_diffuse[~mask] = bg

    show_and_save(
        I_diffuse,
        "Diffuse (Lambertian) - Directional Light",
        "01_diffuse_lambertian.png",
        cmap="gray",
    )



    # -------------------------
    # 2) Diffuse + distance attenuation (directional light, idealized point light)
    # -------------------------
    light_pos = np.array([2.0, 2.0, 2.0]) # location of the light source
    
    I_dist = point_light_attenuated_diffuse(P, N, light_pos, albedo=albedo)
    I_dist[~mask] = bg


    show_and_save(
        I_dist,
        "Diffuse + Distance Attenuation (Point Light)",
        "02_diffuse_distance.png",
        cmap="gray",
    )


    # -------------------------
    # 3) Diffuse + specular (Blinn-Phong) with idealized point light
    # -------------------------
    
    light_pos = np.array([2.0, 2.0, 2.0]) # location of the light source
    view_dir = np.array([0.0, 0.0, 1.0])  # camera looking along +z
    

    x = 50 # Material property that expresses the amount of surface shininess
    s = 0.5 # Specular reflectance 


    I_spec = blinn_phong_specular(P, N, light_pos, view_dir, s=s, x=x)

    I_phong = I_dist + I_spec
    
    I_phong[~mask] = bg


    show_and_save(
        I_phong,
        f"Diffuse + Specular (Blinn-Phong)",
        "03_diffuse_specular.png",
        cmap="gray",
    )


if __name__ == "__main__":
    main()