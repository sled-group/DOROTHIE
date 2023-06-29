
# Co_Wizard 

Co-Wizard controls the agent’s behaviors and carries out language communication with the human participant to jointly achieve a goal. 

## Description of interface

The left shows the driver's first person view The right is a map view of the whole town, while some places like home and person is not shown.

## Co_Wizard high-level physical actions
Each action is mapped to a rule-based local trajectory planner to generate a list of waypoints that the vehicle will drive through.
- T: push-to-talk key. For communicating with Dorothy.
- A: left lane change
- D: right lane change
- S: engage manual brake
- R: release manual brake
- F: enter/exit "parked" state (same as manual brake but with "Parked" overlay). 
- U: attempt U-turn
- up/down keys: increase/decrease speed by 5km/h
- Left click on certain red dot on the map view: turn to the road represented by the dot
  


## Co_Wizard belief tracking action
Co-Wizard annotates the intended actions during and after the inter actions. Human participants can annotate by left clicking on a set of intersections to annotate a set of Jturn actions
- P: start/exit the belief annotation mode
- Z: abandon all plans now (usually when the plan is updated)
- Left click on intersections on the map view: plan to arrive at this intersection, sequentially 
- Right click on anywhere on the map view: set this place as subgoal and plan to arrive here 
  


## Description of executables:

- co_wizard.py: Co_Wizard of Oz executable. Semi-autonomous navigation, user decides direction at intersection, speed, and lane changes.
- co_wizard_map.py: 2D "google maps" view for co_wizard. Connects to CARLA server and to co_wizard.py via LCM
- co_wizard_agent.py: A physical action control module that receive commands from co_wizard and interact with the local planner
- co_wizard_local_planner.py: Modified from carla local_planner for waypoint navigation and used to do the PID control of the agent
- co_wizard_speech_server.py: The speech server used to record and recognize the speech of the human participant
- config_gen.py: Generate a random configuration and storyboard based on the given metaconfig and template