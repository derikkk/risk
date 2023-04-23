import os
import random
from collections import namedtuple, deque

import heapdict
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

import risk.definitions

Territory = namedtuple('Territory', ['territory_id', 'player_id', 'armies'])
Move = namedtuple('Attack', ['from_territory_id', 'from_armies', 'to_territory_id', 'to_player_id', 'to_armies'])


class Board(object):
    """
    The Board object keeps track of all armies situated on the Risk
    world map. Through the definitions it knows the locations of and
    connections between all territories. It handles ownership, attacks
    and movements of armies.

    Args:
        data (list): a sorted list of tuples describing the state of the
            board, each containing three values:
            - tid (int): the territory id of a territory,
            - pid (int): the player id of the owner of the territory,
            - n_armies (int): the number of armies on the territory.
            The list is sorted by the tid, and should be complete.
    """

    def __init__(self, data):
        self.data = data

    @classmethod
    def create(cls, n_players):
        """
        Create a Board and randomly allocate thi
        e territories. Place one army on each territory.

        Args:
            n_players (int): Number of players.

        Returns:
            Board: A board with territories randomly allocated to the players.
        """
        allocation = (list(range(n_players)) * 42)[0:42]
        random.shuffle(allocation)
        return cls([Territory(territory_id=tid, player_id=pid, armies=1)
                    for tid, pid in enumerate(allocation)])

    # ====================== #
    # == Neighbor Methods == #
    # ====================== #

    def neighbors(self, territory_id):
        """
        Create a generator of all territories neighboring a given territory.
 
        Args:
            territory_id (int): ID of the territory to find neighbors of.

        Returns:
            generator: Generator of Territories.
        """
        neighbor_ids = risk.definitions.territory_neighbors[territory_id]
        return (t for t in self.data if t.territory_id in neighbor_ids)

    def hostile_neighbors(self, territory_id):
        """
        Create a generator of all territorie
        s neighboring a given territory, of which
        the owner is not the same as the owner of the original territory.

        Args:
            territory_id (int): ID of the territory.

        Returns:
            generator: Generator of Territories.
        """
        player_id = self.owner(territory_id)
        neighbor_ids = risk.definitions.territory_neighbors[territory_id]
        return (t for t in self.data if (t.player_id != player_id and t.territory_id in neighbor_ids))

    def friendly_neighbors(self, territory_id):
        """
        Create a generator of all terr
        itories neighboring a given territory, of which
        the owner is the same as the owner of the original territory.

        Args:
            territory_id (int): ID of the territory.

        Returns:
            generator: Generator of tuple
            s of the form (territory_id, player_id, armies).
        """
        player_id = self.owner(territory_id)
        neighbor_ids = risk.definitions.territory_neighbors[territory_id]
        return (t for t in self.data if (t.player_id == player_id and t.territory_id in neighbor_ids))


    # ================== #
    # == Path Methods == #
    # ================== #

    def is_valid_path(self, path):
        '''
        A path is list of territories satisfying two conditions:
        1. For all territories V in the lis
        t (except the last one), the next territory
        W is in the neighbors if V.
        2. No territory is repeated multiple times.
        Valid paths can be of any length (including 0 and 1).

        Args:
            path ([int]): a list of territory_ids which represent the path

        Returns:
            bool: True if the input path is valid
        '''
        s = set()
        if len(path) <= 1:
            return True
        else:
            for i in range(0, len(path) - 1):
                first = path[i]
                second = path[i+1]
                s.add(first)
                s.add(second)
                if second not in risk.definitions.territory_neighbors[first]:
                    return False
            if len(s) != len(path):
                return False
            return True
    
    def is_valid_attack_path(self, path):
        '''
        The rules of Risk state that when attacking
        a player's armies cannot move through terr
        itories they already occupy;
        they must move through enemy territories.
        All valid attacks, therefore, will follow
        a path of starting on one player's territor
        y and moving trough enemy territories.

        Formally, an attack path is a valid pat
        h satisfying the following two additional properties:
        1. An attack path must contain at least two territories;
        1. If the first territory is owned by playe
        r A, then no other territories in the path are also owned by A.

        Args:
            path ([int]): a list of territory_ids which represent the path

        Returns:
            bool: True if the path is an attack path
        '''
        if not self.is_valid_path(path) or len(path) < 2:
            return False
        else:
            owner = self.owner(path[0])
            for i in path[1:]:
                if self.owner(i) == owner:
                    return False
            return True

    def cost_of_attack_path(self, path):
        count = 0
        for i in path[1:]:
            count += self.armies(i)
        return count

    def shortest_path(self, source, target):
        dictt = {}
        dictt[source] = [source]
        q = deque()
        q.append(source)
        visited = set()
        visited.add(source)

        while q:
            current_ter = q.popleft()
            if current_ter == target:
                return dictt[current_ter]
            for ter in risk.definitions.territory_neighbors[current_ter]:
                if ter not in visited:
                    new_ter = copy.deepcopy(dictt[current_ter])
                    new_ter.append(ter)
                    dictt[ter] = new_ter
                    q.append(ter)
                visited.add(ter)

    def can_fortify(self, source, target):
        path = {source: [source]}
        queue = deque([source])
        visited = {source}

        while queue:
            current = queue.popleft()
            if current == target:
                return True
            for territory in risk.definitions.territory_neighbors[current]:
                if territory in visited or self.owner(territory) != self.owner(source):
                    continue
                temp_dict = path[current] + [territory]
                if territory in path and len(temp_dict) >= len(path[territory]):
                    continue
                path[territory] = temp_dict
                queue.append(territory)
            visited.add(current)
        return False


    def cheapest_attack_path(self, source, target):
        if self.owner(source) == self.owner(target):
            return None
        path = {source: [source]}
        queue = heapdict.heapdict()
        queue[source] = 0
        visited = set()
        visited.add(source)
        
        while queue:
            current, priority = queue.peekitem()
            queue.pop(current)
            if current == target:
                return path[current]
            for neighbor in risk.definitions.territory_neighbors[current]:
                if neighbor in visited or self.owner(neighbor) == self.owner(source):
                    continue
                temp_path = path[current].copy()
                temp_path.append(neighbor)
                neighbor_priority = priority + self.armies(neighbor)
                if neighbor not in path or neighbor_priority < queue.get(neighbor, float('inf')):
                    path[neighbor] = temp_path
                    queue[neighbor] = neighbor_priority
            visited.add(current)
        return None

    def can_attack(self, source, target):
        '''
        Args:
            source (int): territory_id of source node
            target (int): territory_id of target node

        '''
        return self.cheapest_attack_path(source, target)

    # ======================= #
    # == Continent Methods == #
    # ======================= #

    def continent(self, continent_id):
        """
        Args:
            continent_id (int): ID of the continent.

        Returns:
            generator: Generator of Territories.
        """
        return (t for t in self.data if t.territory_id in risk.definitions.continent_territories[continent_id])

    def n_continents(self, player_id):
        """
        Calculate the total number of continents owned by a player.
        
        Args:
            player_id (int): ID of the player.
                
        Returns:
            int: Number of continents owned by the player.
        """
        return len([continent_id for continent_id in range(6) if self.owns_continent(player_id, continent_id)])

    def owns_continent(self, player_id, continent_id):
        """
        Check if a player owns a continent.
        
        Args:
            player_id (int): ID of the player.
            continent_id (int): ID of the continent.
            
        Returns:
            bool: True if the player owns all of the continent's territories.
        """
        return all((t.player_id == player_id for t in self.continent(continent_id)))

    def continent_owner(self, continent_id):
        """
        Find the owner of all territories in a continent. If the continent
        is owned by various players, return None.
            
        Args:
            continent_id (int): ID of the continent.
                
        Returns:
            int/None: Player_id if a player owns all territories, else None.
        """
        pids = set([t.player_id for t in self.continent(continent_id)])
        if len(pids) == 1:
            return pids.pop()
        return None

    def continent_fraction(self, continent_id, player_id):
        """
        Compute the fraction of a continent a player owns.
        
        Args:
            continent_id (int): ID of the continent.
            player_id (int): ID of the player.

        Returns:
            float: The fraction of the continent owned by the player.
        """
        c_data = list(self.continent(continent_id))
        p_data = [t for t in c_data if t.player_id == player_id]
        return float(len(p_data)) / len(c_data)

    def num_foreign_continent_territories(self, continent_id, player_id):
        """
        Compute the number of territories owned by other players on a given continent.
        
        Args:
            continent_id (int): ID of the continent.
            player_id (int): ID of the player.
        """
        return sum(1 if t.player_id != player_id else 0 for t in self.continent(continent_id))

    # ==================== #
    # == Action Methods == #
    # ==================== #    

    def reinforcements(self, player_id):
        """
        Calculate the number of reinforcements a player is entitled to.
            
        Args:
            player_id (int): ID of the player.

        """
        base_reinforcements = max(3, int(self.n_territories(player_id) / 3))
        bonus_reinforcements = 0
        for continent_id, bonus in risk.definitions.continent_bonuses.items():
            if self.continent_owner(continent_id) == player_id:
                bonus_reinforcements += bonus
        return base_reinforcements + bonus_reinforcements

    def possible_attacks(self, player_id):
        """
        Assemble a list of all possible attacks for the players.

        Args:
            player_id (int): ID of the attacking player.

        Returns:
            list: List of Moves.
        """
        return [Move(from_t.territory_id, from_t.armies, to_t.territory_id, to_t.player_id, to_t.armies)
                for from_t in self.mobile(player_id) for to_t in self.hostile_neighbors(from_t.territory_id)]

    def possible_fortifications(self, player_id):
        """
        Assemble a list of all possible fortifications for the players.
        
        Args:
            player_id (int): ID of the attacking player.

        Returns:
            list: List of Moves.
        """
        return [Move(from_t.territory_id, from_t.armies, to_t.territory_id, to_t.player_id, to_t.armies)
                for from_t in self.mobile(player_id) for to_t in self.friendly_neighbors(from_t.territory_id)]

    def fortify(self, from_territory, to_territory, n_armies):
        """
        Perform a fortification.
        """
        if n_armies < 0 or self.armies(from_territory) <= n_armies:
            raise ValueError('Board: Cannot move {n} armies from territory {tid}.'
                             .format(n=n_armies, tid=from_territory))
        if to_territory not in [t.territory_id for t in self.friendly_neighbors(from_territory)]:
            raise ValueError('Board: Cannot fortify, territories do not share owner and/or border.')
        self.add_armies(from_territory, -n_armies)
        self.add_armies(to_territory, +n_armies)

    def attack(self, from_territory, to_territory, attackers):
        """
        Perform an attack.

        Args:
            from_territory (int): Territory_id of the offensive territory.
            to_territory (int): Territory_id of the defensive territory.
            attackers (int): Number of attacking armies.
        """
        if attackers < 1 or self.armies(from_territory) <= attackers:
            raise ValueError('Board: Cannot attack with {n} armies from territory {tid}.'
                             .format(n=attackers, tid=from_territory))
        if to_territory not in [tid for (tid, _, _) in self.hostile_neighbors(from_territory)]:
            raise ValueError('Board: Cannot attack, territories do not share border or are owned by the same player.')
        defenders = self.armies(to_territory)
        def_wins, att_wins = self.fight(attackers, defenders)
        if self.armies(to_territory) == att_wins:
            self.add_armies(from_territory, -attackers)
            self.set_armies(to_territory, attackers - def_wins)
            self.set_owner(to_territory, self.owner(from_territory))
            return True
        else:
            self.add_armies(from_territory, -def_wins)
            self.add_armies(to_territory, -att_wins)
            return False

    # ====================== #
    # == Plotting Methods == #
    # ====================== #    

    def plot_board(self, path=None, plot_graph=False, filename=None):
        """ 
        Plot the board. 
        
        Args:
        """
        im = plt.imread(os.getcwd() + '/img/risk.png')
        dpi=96
        img_width=800
        fig, ax = plt.subplots(figsize=(img_width/dpi, 300/dpi), dpi=dpi)
        _ = plt.imshow(im)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        def plot_path(xs):
            if not self.is_valid_path(xs):
                print('WARNING: not a valid path')
            coor = risk.definitions.territory_locations[xs[0]]
            verts=[(coor[0]*1.2, coor[1]*1.22 + 25)]
            codes = [ Path.MOVETO ]
            for i,x in enumerate(xs[1:]):
                if (xs[i]==19 and xs[i+1]==1) or (xs[i]==1 and xs[i+1]==19):
                    coor = risk.definitions.territory_locations[x]
                    #verts.append((coor[0]*1.2, coor[1]*1.22 + 25))
                    verts.append((1000,-200))
                    verts.append((coor[0]*1.2, coor[1]*1.22 + 25))
                    codes.append(Path.CURVE3)
                    codes.append(Path.CURVE3)
                else:
                    coor = risk.definitions.territory_locations[x]
                    verts.append((coor[0]*1.2, coor[1]*1.22 + 25))
                    codes.append(Path.LINETO)
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=2)
            ax.add_patch(patch)

        if path is not None:
            plot_path(path)

        if plot_graph:
            for t in risk.definitions.territory_neighbors:
                path = []
                for n in risk.definitions.territory_neighbors[t]:
                    path.append(t)
                    path.append(n)
                plot_path(path)

        for t in self.data:
            self.plot_single(t.territory_id, t.player_id, t.armies)

        if not filename:
            plt.tight_layout()
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(filename,bbox_inches='tight')

    @staticmethod
    def plot_single(territory_id, player_id, armies):
        """
        Plot a single army dot.
            
        Args:
            territory_id (int): the id of the territory to plot,
            player_id (int): the player id of the owner,
            armies (int): the number of armies.
        """
        coor = risk.definitions.territory_locations[territory_id]
        plt.scatter(
            [coor[0]*1.2], 
            [coor[1]*1.22], 
            s=400, 
            c=risk.definitions.player_colors[player_id],
            zorder=2
            )
        plt.text(
            coor[0]*1.2, 
            coor[1]*1.22 + 15, 
            s=str(armies),
            color='black' if risk.definitions.player_colors[player_id] in ['yellow', 'pink'] else 'white',
            ha='center', 
            size=15
            )

    # ==================== #
    # == Combat Methods == #
    # ==================== #    

    @classmethod
    def fight(cls, attackers, defenders):
        """
        Stage a fight.

        Args:
            attackers (int): Number of attackers.
            defenders (int): Number of defenders.

        """
        n_attack_dices = min(attackers, 3)
        n_defend_dices = min(defenders, 2)
        attack_dices = sorted([cls.throw_dice() for _ in range(n_attack_dices)], reverse=True)
        defend_dices = sorted([cls.throw_dice() for _ in range(n_defend_dices)], reverse=True)
        wins = [att_d > def_d for att_d, def_d in zip(attack_dices, defend_dices)]
        return len([w for w in wins if w is False]), len([w for w in wins if w is True])

    @staticmethod
    def throw_dice():
        """
        Throw a dice.
        
        Returns:
            int: random int in [1, 6]. """
        return random.randint(1, 6)

    # ======================= #
    # == Territory Methods == #
    # ======================= #

    def owner(self, territory_id):
        """
        Get the owner of the territory.

        Args:
            territory_id (int): ID of the territory.

        Returns:
            int: Player_id that owns the territory.
        """
        return self.data[territory_id].player_id

    def armies(self, territory_id):
        """
        Get the number of armies on the territory.

        Args:
            territory_id (int): ID of the territory.

        Returns:
            int: Number of armies in the territory.
        """
        return self.data[territory_id].armies

    def set_owner(self, territory_id, player_id):
        """
        Set the owner of the territory.

        Args:
            territory_id (int): ID of the territory.
            player_id (int): ID of the player.
        """
        self.data[territory_id] = Territory(territory_id, player_id, self.armies(territory_id))

    def set_armies(self, territory_id, n):
        """
        Set the number of armies on the territory.

        Args:
            territory_id (int): ID of the territory.
            n (int): Number of armies on the territory.

        Raises:
            ValueError if n < 1.
        """
        if n < 1:
            raise ValueError('Board: cannot set the number of armies to <1 ({tid}, {n}).'.format(tid=territory_id, n=n))
        self.data[territory_id] = Territory(territory_id, self.owner(territory_id), n)

    def add_armies(self, territory_id, n):
        """
        Add (or remove) armies to/from the territory.

        Args:
            territory_id (int): ID of the territory.
            n (int): Number of armies to add to the territory.

        Raises:
            ValueError if the resulting number of armies is <1.
        """
        self.set_armies(territory_id, self.armies(territory_id) + n)

    def n_armies(self, player_id):
        """
        Count the total number of armies owned by a player.

        Args:
            player_id (int): ID of the player.

        Returns:
            int: Number of armies owned by the player.
        """
        return sum((t.armies for t in self.data if t.player_id == player_id))

    def n_territories(self, player_id):
        """
        Count the total number of territories owned by a player.

        Args:
            player_id (int): ID of the player.

        Returns:
            int: Number of territories owned by the player.
        """
        return len([None for t in self.data if t.player_id == player_id])

    def territories_of(self, player_id):
        """
        Return a set of all territories owned by the player.

        Args:
            player_id (int): ID of the player.

        Returns:
            list: List of all territory IDs owner by the player.
        """
        return [t.territory_id for t in self.data if t.player_id == player_id]

    def mobile(self, player_id):
        """
        Args:
            player_id (int): ID of the attacking player.

        Returns:
            generator: Generator of Territories.
        """
        return (t for t in self.data if (t.player_id == player_id and t.armies > 1))
