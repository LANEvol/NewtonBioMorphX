
<div align="center">
<img src="docs/images/nbmx_banner_sm.png" alt="NewtonBioMorphX banner" width="800">
</div>

# NewtonBioMorphX

**A GPU-accelerated framework for mechanical morphogenesis**

NewtonBioMorphX is an open-source application for simulating the tissue-scale mechanics of morphogenesis. Directly from a graphical interface, you can generate geometries, define material properties, prescribe growth, add anisotropic fiber reinforcement and active forces, visualize results in real time, and export simulations for further analysis, without editing or recompiling any code.

It is designed for biologists and physicists alike, and is also a powerful teaching tool for exploring the principles of mechanical patterning hands-on.

<div align="center">
<img src="docs/images/main_interface.png" alt="NewtonBioMorphX user interface" width="900">
<p><em>The control panel (left) and the real-time rendering window (right).</em></p>
</div>

## Features

- **Geometry:** built-in multilayer mesh generator (box, disk, tube, cone, sphere, torus), or import your own VTK mesh
- **Materials:** elasticity, viscosity and plasticity, defined per layer, with optional rigid layers
- **Growth:** isotropic or anisotropic growth rates, varying in space and time
- **Anisotropy:** up to four passive fiber families and active tensile forces per layer
- **Coordinate systems:** normal/tangent, Cartesian, cylindrical, cone-adapted, spherical and toroidal
- **Boundary conditions:** free, traction, fixed and tangential slide
- **Analytical expressions** for every parameter, compiled at runtime into GPU kernels
- **Robust solver:** contact handling, F-bar formulation for nearly incompressible materials, automatic recovery from numerical instabilities
- **Visualization:** surface, section and clip views, scalar fields, fiber and growth directions, mesh-quality histogram
- **Export:** VTK, PLY and MP4 movies

## Examples

Twelve ready-to-run presets are included. A few of them:

<table>
<tr>
<td align="center"><img src="docs/images/preset_1.png" width="250"><br><sub>Elastic box with tangential growth</sub></td>
<td align="center"><img src="docs/images/preset_5.png" width="250"><br><sub>Tube with growth and collagen fibers</sub></td>
<td align="center"><img src="docs/images/preset_8.png" width="250"><br><sub>Two-layer sphere with orthogonal growth</sub></td>
<td align="center"><img src="docs/images/preset_12.png" width="250"><br><sub>Elastic torus with fibers</sub></td>
</tr>
</table>

See the [full list of presets](docs/User_Manual.md#5-preset-examples).

## Download

Ready-to-use packages are available on the [Releases page](https://github.com/LANEvol/NewtonBioMorphX/releases):

| Operating system | File | Description |
|------------------|------|-------------|
| Windows | `NewtonBioMorphX-<version>-win64.exe` | Installer (recommended) |
| Windows | `NewtonBioMorphX-<version>-win64.zip` | Portable version, no installation required |
| Linux | `NewtonBioMorphX-<version>-x86_64.AppImage` | Single-file executable, no installation required |

**Requirements:** an NVIDIA GPU with an up-to-date driver, and 64-bit Windows 10/11 or Linux (e.g., Ubuntu 22.04 or newer). The CUDA Toolkit is **not** required to run the application.

On Linux, make the AppImage executable, then launch it:

```
chmod +x NewtonBioMorphX-<version>-x86_64.AppImage
./NewtonBioMorphX-<version>-x86_64.AppImage
```

## Documentation

📖 **[User Manual](docs/User_Manual.md)**: installation, interface, modeling workflow, expression syntax, simulation parameters, visualization, and FAQ.

## Building from source

NewtonBioMorphX is built with CMake (3.21 or newer) on Windows (Visual Studio 2022 + vcpkg) and on Linux (Docker). See [Compiling source code](docs/User_Manual.md#3-compiling-source-code) in the manual.

## Citation

If you use NewtonBioMorphX in your research, please cite:

> Ebrahim Jahanbakhsh, Michel C. Milinkovitch, NewtonBioMorphX: a computational framework for simulating tissue-scale mechanics of morphogenesis, Computer Physics Communications (2026), https://doi.org/10.1016/j.cpc.2026.110452.

## License

NewtonBioMorphX is free software, released under the [GNU General Public License v3.0](LICENSE).

## Authors

Ebrahim Jahanbakhsh & Michel C. Milinkovitch
[Laboratory of Artificial and Natural Evolution (LANE)](https://www.lanevol.org/), University of Geneva

Project website: https://www.lanevol.org/projects/NewtonBioMorphX