# CrateSlidingRL
This is for our Final Project (CICS 687) - CrateSliding Domain

### Problem Statement

You are given a grid-based environment representing a room. The environment contains various elements: pitchforks (in red), crates (in blue), obstacles (in black) and a target state (in green). The goal is to navigate the pitchfork in the fewest possible steps to the target location. Each cell within the grid can accommodate only one entity (either a pitchfork or a crate). Cells with obstacles cannot accommodate any entity. Movement within the grid is limited to four directions: up, down, right, left. Any pitchfork or crate can be moved in this direction to an adjacent cell as long as it is not obstructed by an obstacle or occupied by another entity. Our objective is to optimize the pathfinding strategy to minimize the number of steps taken to guide the pitchfork to the target location while avoiding obstacles and crates.
