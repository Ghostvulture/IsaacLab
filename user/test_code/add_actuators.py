"""This script demonstrates how to import a robot and ground from USD files.

Usage:
    ./isaaclab.sh -p user/test_code/import.py
"""

import argparse

from isaaclab.app import AppLauncher

# Create argparser
parser = argparse.ArgumentParser(description="Import robot and ground from USD files.")
# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils


def design_scene() -> list[list[float]]:
    """Designs the scene by spawning ground plane and robots."""
    
    # Ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # Lighting
    cfg_light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light)
    
    # Get the path to the USD file
    # Using the COD-2026RoboMaster-Balance.usd file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    usd_path = os.path.join(current_dir, "../usd_file/USD/COD-2026RoboMaster-Balance.usd")
    
    origin = [0.0, 0.0, 0.0]
    # Origin
    prim_utils.create_prim("/World/Origin", "Xform", translation=origin)

    # Spawn robots from USD file
    cfg_robot = sim_utils.UsdFileCfg(usd_path=usd_path)
    cfg_robot.func("/World/Origin/Robot", cfg_robot, translation=(0.0, 0.0, 0.0))
    
    # Return the origins
    return origin

def run_simulators():
    


def main():
    """Main function."""
    
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main camera view (eye position, target position)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.5])
    
    # Design scene
    scene_origins = design_scene()
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print("[INFO]: Robot loaded successfully!")
    print("[INFO]: Robot at position:", scene_origins)
    
    # Simulation loop
    while simulation_app.is_running():
        # Perform simulation step
        sim.step()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
