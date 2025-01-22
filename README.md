# Hybrid Optimization of Horizontal Alignments

This repository contains the Python code used in the paper **Hybrid Optimization of Horizontal Alignments in European Terrains: A Comparative Study** by Ane Espeseth, Martin Juříček, Harald M. Ludwig and Tea Tušar accepted for publication at [Evostar 2025](https://www.evostar.org/2025/).

## Easy Setup
### Docker

```
docker build -t your_name_of_docker .
docker run -it --rm your_name_of_docker bash
python3 scripts/main.py
```

### Configuration

The `main_conf` dictionary in  `main.py` defines the main configuration parameters for the optimization run. Below is the detailed description of each parameter:

| **Key**                  | **Type**       | **Description**                                                                 | **Default/Example**                   |
|--------------------------|----------------|---------------------------------------------------------------------------------|---------------------------------------|
| `heightmap`              | `string`       | Path to the selected heightmap (from `conf_example`).                           | Example: `"./heightmaps/slovenia.npy"`|
| `res`                    | `(int, int)`   | Resolution for the selected heightmap.                                          | Example: `(200, 400)`                 |
| `start`                  | `(int, int)`   | Start coordinates (from `conf_example`).                                        | Example: `(175, 0)`                   |
| `goal`                   | `(int, int)`   | Goal coordinates (from `conf_example`).                                         | Example: `(160, 290)`                 |
| `algo`                   | `string`       | Selected optimization algorithm (from `conf_example["opti_algos"]`).            | Example: `"CMAES"`                    |
| `cutting_plane_factor`   | `float`        | Cutting plane factor used in optimization.                                      | Default: `1`                          |
| `epsilon`                | `float`        | Epsilon parameter for optimization.                                             | Default: `1`                          |
| `pop_size`               | `int`          | Population size for the optimization algorithm.                                 | Default: `60`                         |
| `gen_size`               | `int`          | Number of generations for the optimization algorithm.                           | Default: `50`                         |
| `tau`                    | `float`        | Tau parameter for optimization.                                                 | Default: `0.4`                        |
| `seed`                   | `int`          | Random seed for reproducibility.                                                | Example: `10`                         |
| `run_name`               | `string`       | Unique identifier for the run, typically generated using `uuid`.                | Example: `"f47ac10b58cc4372a5670e02"` |


## Paper Abstract
<p align="justify">Path planning across terrain is a fundamental challenge in civil engineering, with applications ranging from transportation infrastructure to urban development. Recent advances in computational methods have enabled automated route optimization, particularly in horizontal alignment problems that balance construction costs with terrain constraints. However, standardized comparisons of optimization approaches across diverse geographical contexts remain limited, hindering the development of reliable automated planning systems. Here we show through a systematic comparative study across three European landscapes that A* significantly outperforms RRT* in initial path generation, with better computational efficiency and terrain adaptation, while PSO demonstrates superior optimization capabilities compared to CMA-ES and DE in refining these paths against roadway construction criteria. Through extensive parameter validation, we find these performance advantages remain consistent across different geographical contexts and topographical challenges, with the hybrid A*-PSO approach achieving significantly better results than applying optimization algorithms to straight-line paths alone. These findings provide a comprehensive comparison of key algorithms in infrastructure planning optimization, demonstrating the relative strengths of different approaches in horizontal alignment tasks. This comparative analysis offers practical guidance for algorithm selection while highlighting opportunities for further development through the incorporation of real-world engineering constraints.</p>

## Examples

<table>
  <tr>
    <td>Comparison between paths found by A* and RRT* on the Slovenian map.</td>
    <td>Comparison between the path found by A* and its further optimized trajectory on the Austrian map.</td>
  </tr>
  <tr>
    <td><img src="docs/ex3.png"></td>
    <td><img src="docs/ex2.png"></td>
  </tr>
</table>

## Authors

* Ane Espeseth 📫 ane.espeseth@gmail.com
* Martin Juricek 📫 200543@vutbr.cz
* Harald M. Ludwig 📫 ludwig@csh.ac.at
* Tea Tušar





