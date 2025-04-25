from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from space_exploration.exploration.shipyard import Shipyard


class LaunchPad:
    def __init__(self, shipyard: 'Shipyard'):
        self.shipyard = shipyard



if __name__ == '__main__':
    shipyard = Shipyard()
    launchpad = LaunchPad()


