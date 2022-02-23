from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio.types import Mission, Route
from smarts.sstudio import gen_traffic, gen_missions

scenario = str(Path(__file__).parent)


agent_missions = [
<<<<<<< HEAD
    Mission(Route(begin=("edge-west-WE", 0, 80), end=("edge-east-WE", (1,), 20))),
    Mission(Route(begin=("edge-west-WE", 1, 60), end=("edge-east-WE", (1,), 20))),
    #Mission(Route(begin=("edge-east-EW", 0, 80), end=("edge-south-NS", (1,), 40))),
    Mission(Route(begin=("edge-south-SN", 1, 60), end=("edge-west-EW", (1,), 20))),
    Mission(Route(begin=("edge-south-SN", 1, 80), end=("edge-west-EW", (1,), 20))),
    Mission(Route(begin=("edge-south-SN", 0, 80), end=("edge-east-WE", (0,), 20))),
    Mission(Route(begin=("edge-south-SN", 0, 60), end=("edge-east-WE", (1,), 20))),
    
=======
    Mission(Route(begin=("edge-west-WE", 0, 80), end=("edge-east-WE", (1,), 40))),
    Mission(Route(begin=("edge-east-EW", 0, 80), end=("edge-south-NS", (1,), 40))),
    Mission(Route(begin=("edge-south-SN", 0, 80), end=("edge-west-EW", (1,), 40))),
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09
]

gen_missions(scenario, missions=agent_missions, overwrite=True)


traffic = t.Traffic(
    flows=[
        t.Flow(
<<<<<<< HEAD
            route=t.Route(begin=("edge-west-WE", 0, 10), end=("edge-east-WE", 0, -10),),#
            rate=2400,
=======
            route=t.Route(begin=("edge-west-WE", 0, 10), end=("edge-east-WE", 0, -10),),
            rate=400,
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09
            actors={t.TrafficActor("car"): 1,},
        ),
        t.Flow(
            route=t.Route(begin=("edge-east-EW", 0, 10), end=("edge-west-EW", 0, -10),),
<<<<<<< HEAD
            rate=2400,
=======
            rate=400,
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09
            actors={t.TrafficActor("car"): 1,},
        ),
    ]
)

gen_traffic(scenario, traffic, name="basic", overwrite=True)
