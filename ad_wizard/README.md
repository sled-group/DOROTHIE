# Ad_Wizard

Ad_wizard introduce unexpected situations, controllers can add environmental exceptions and task exceptions

## Description of interface

The left shows the bird-eye view of the car. The upper right is a map view of the whole town and the trajectory of the car, and can also be clicked to add exceptions. The bottom right are weather controlling buttons and can be toggled or clicked to control weather conditions like fog and rains.

## Environmental Exceptions

### Typed Environmental Exceptions 
To enter a command, press space followed by the command, then enter. Some commands have multiple stages (see below).
- Place prop. Type colon followed by the name of the prop (names are defined at the top of ad_wizard.py). Then click location on 2D map to place object, then click again to indicate orientation (object will face in direction towards second click). Full example: type space followed by ":trafficcone", then press enter. Click twice to indicate location and orientation.
  
  possible assets to be added: tree, constructioncone, dumpster, streetbarrier, trafficcone, trafficsign
- Send message to Dorothy. Enter '@' followed by the message, then enter. Full example: type space followed by '@Hello Dorothy' then enter. e.g.: xxx street is closed.
- Spawning clusters of NPC people and cars. Type '!people' or '!cars', optionally followed by the number to spawn (default 10). Then subsequently click two corners of the rectangular area to spawn the cars/people in.

### Other Environmental Exceptions (not recommended)
- Delete prop: right-click thumbnail on map.
- Move prop: left-click and hold prop to drag to new location.

### Weather controls
The ad_wizard controls the weather parameters using the various slide controls at the bottom of the panel. Additionally, left-clicking a parameter will decrease it by 10, and right-clicking will conversely increase.

You can also toggle the panel to left or right to change the weather parameters


## Task Exceptions

- Trigger a task: type in t and enter. It will show the task triggered if in the template specified the "trigger" option of a subgoal is true,

- Delete a task: type in d and enter. It will show the task deleted if in the template specified the "delete" option of a subgoal is true,

- change a task: type in c and enter. It will show the task changed if in the template specified the "change" option of a subgoal is true,

because it only change the task description on dorothy's view, usually ad_wizard will also send a sentence starting by '@' to tell human participant what happened.

## Description of executables:
- ad_wizard/ad_wizard.py: main ad_wizard executable. Connects to CARLA server and to Dorothy via tunneling through co_wizard (soon to be changed to streamline networking).
- dorothy/dorothy.py: main Dorothy executable. Shows 3D view of scene without overlays (except messages from ad_wizard). 
- dorothy/dorothy_map.py: 2D view for dorothy, stripped-down version of co_wizard.
