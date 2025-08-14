from itertools import combinations

def calc_shapley_value(player_index, all_players, cf_dict):
    """
    Calculate the Shapley value for player index
    Input:
        player_index: index of player to calculate Shapley value of
        all_players: list of player indices
        cf_dict: dictionary containing characteristic function values for all players
    """
    players = all_players.copy()

    if player_index in players:
        players.remove(player_index)

    num_players = len(players)
    coalition_sizes = list(range(num_players+1)) 
    value = 0 
    player_tuple = (player_index,) 

    for _size in coalition_sizes: 
        coalition_value = 0
        coalitions_of_size_s = list(combinations(players, _size))
        for _coalition in coalitions_of_size_s:
            value_in_coalition = (cf_dict[tuple(sorted(_coalition + player_tuple))] - cf_dict[_coalition]) 
            coalition_value += value_in_coalition

        average_coalition_value = coalition_value/len(coalitions_of_size_s)
        value += average_coalition_value
    average_value = value/len(coalition_sizes)

    return average_value

def calc_shapley_values(x, cf_dict):
    """
    Returns the shapley values for features x, given a
    characteristic function dictionary
    """
    
    players = list(range(x.shape[1]))
    shapley_values = []
    for _player in players:
        shapley_values.append(calc_shapley_value(_player, players, cf_dict))
    return shapley_values

def flatten(tup):
    """ 
    Flatten any nested tuple
    """
    if len(tup) < 1:
        return tup 
    if isinstance(tup[0], tuple):
        return flatten(tup[0]) + flatten(tup[1:])
    return tup[:1] + flatten(tup[1:])


def make_cf_dict(x, y, characteristic_function):
    """ 
    Creates dictionary with values of the characteristic function for each
    combination of the players.
    """
    cf_dict = {}
    num_players = x.shape[1]
    players = list(range(num_players))
    coalition_sizes = list(range(num_players+1))

    for _size in coalition_sizes:
        coalitions_of_size_s = list(combinations(players, _size))
        for _coalition in coalitions_of_size_s:
            _coalition = tuple(sorted(flatten(_coalition)))
            cf_dict[_coalition] = characteristic_function(x, y, _coalition)

    return cf_dict

if __name__=="__main__":
    players = [1,2,3]
    cf_dict = {():0, (1,):3, (2,):7, (3,):10, (1,2):7, (1,3):10, (2,3):10, (1,2,3):10}
    print(calc_shapley_value(1, players, cf_dict))
    print(calc_shapley_value(2, players, cf_dict))
    print(calc_shapley_value(3, players, cf_dict))

