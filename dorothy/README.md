# Dorothy
dorothy is the interface that the human participant in to duo-wizard setup will see. 

## Description of interface
In bottom left there will be a list of subgoals to accomplish. All subgoals except the last one can be performed not in the same order listed, but the last subgoal must be reached last. The humane participant is required to collaborate with the co_wizard to navigate through the city and reach the subgoals.

The left shows the driver's first person view and is also what co_wizard can see. The right is a map view of the whole town.

## Description of executables:
- dorothy.py: main Dorothy executable. Shows 3D view of scene without overlays (except messages from ad_wizard). 
- dorothy_map.py: 2D view for dorothy, stripped-down version of co_wizard.
- dorothy_speech_client.py: Speech recognizer for human participant.