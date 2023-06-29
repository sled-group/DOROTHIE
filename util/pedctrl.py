import carla
import random
import logging
from carla import VehicleLightState as vls


def createPeds(client, num_walkers, spawn_rect, sync=False,
               percentagePedestriansRunning=0.0,
               percentagePedestriansCrossing=0.0):

    #spawn rect = (xmin, xmax, ymin, ymax)
    world = client.get_world()
    blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')
    xmin, xmax, ymin, ymax = spawn_rect

    walkers_list = []
    all_id = []

    SpawnActor = carla.command.SpawnActor
    FutureActor = carla.command.FutureActor

    # 1. take all the random locations to spawn
    spawn_points = []
    for i in range(num_walkers):
        spawn_point = carla.Transform()
        spawn_point.location.x = random.uniform(xmin, xmax)
        spawn_point.location.y = random.uniform(ymin, ymax)
        spawn_points.append(spawn_point)

    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)

        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')

        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))

    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2

    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id

    # 4. we put altogether the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    if not sync:
        world.wait_for_tick()
    else:
        world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(carla.Location(random.uniform(xmin, xmax), random.uniform(ymin, ymax), 0))
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))


def createVehicles(client, num_vs, tm_port=8000, spawn_rect=None, sync=False):

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor

    world = client.get_world()
    carlamap = world.get_map()
    traffic_manager = client.get_trafficmanager(tm_port)

    blueprints = world.get_blueprint_library().filter('vehicle.*')

    # --------------
    # Spawn vehicles
    # --------------
    if spawn_rect:
        xmin, xmax, ymin, ymax = spawn_rect
        sps = [p for p in carlamap.get_spawn_points()
               if xmin <= p.location.x <= xmax
               and ymin <= p.location.y <= ymax]
    else:
        sps = carlamap.get_spawn_points()

    if num_vs > len(sps):
        print('NOTE: ad_wizard requested more (%d) vehicles than there are '
              'spawn points in provided bounding box (%d). Rounding down.' % (num_vs, len(sps)))
        num_vs = len(sps)

    batch = []
    for n, transform in enumerate(random.sample(sps, num_vs)):
        #loc_rand = carla.Location(random.uniform(xmin, xmax), random.uniform(ymin, ymax), 0)
        #transform = carlamap.get_waypoint(loc_rand).transform

        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # prepare the light state of the cars to spawn
        light_state = vls.NONE
        #if args.car_lights_on:
        #    light_state = vls.Position | vls.LowBeam | vls.LowBeam

        # spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform)
                     .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                     .then(SetVehicleLightState(FutureActor, light_state)))

    vehicles_list = []
    for response in client.apply_batch_sync(batch, sync):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)

    return vehicles_list
