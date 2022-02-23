import os
<<<<<<< HEAD
import random
=======

>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09
from smarts.sstudio import gen_traffic, gen_missions
from smarts.sstudio.types import (
    Traffic,
    Flow,
    Route,
    RandomRoute,
    TrafficActor,
    Mission,
)

scenario = os.path.dirname(os.path.realpath(__file__))

agent_missions = [
    Mission(Route(begin=("left_in", 1, 60), end=("merged", (0,), 20))),
<<<<<<< HEAD
    #Mission(Route(begin=("left_in", 0, 30), end=("merged", (1,), 40))),
    #Mission(Route(begin=("left_in", 1, 30), end=("merged", (1,), 30))),
    ############# 起始点(车道线 ,车道线编号,位置) 终点(车道线 ,车道线编号,位置)
    Mission(Route(begin=("ramp_in", 0, 10), end=("merged", (1,), 10))),
    #Mission(Route(begin=("ramp_in", 0, 50), end=("merged", (0,), 50))),
    #Mission(Route(begin=("ramp_in", 0, 20), end=("merged", (0,), 50))),
=======
    Mission(Route(begin=("left_in", 0, 30), end=("merged", (1,), 40))),
    Mission(Route(begin=("ramp_in", 0, 30), end=("merged", (0,), 50))),
    Mission(Route(begin=("ramp_in", 0, 50), end=("merged", (0,), 50))),
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09
]

gen_missions(scenario, agent_missions, overwrite=True)

gen_traffic(
    scenario,
    Traffic(
        flows=[
            Flow(
<<<<<<< HEAD
                
               route=RandomRoute(), rate=3600, actors={TrafficActor(name="car"): 1.0},
=======
                route=RandomRoute(), rate=3600, actors={TrafficActor(name="car"): 1.0},
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09
            )
        ]
    ),
    name="random",
    overwrite=True,
)
