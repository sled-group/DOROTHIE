#!/usr/bin/python3

# ==============================================================================
# -- COPYRIGHT -----------------------------------------------------------------
# ==============================================================================

# Copyright (c) 2022 SLED Lab, EECS, University of Michigan.
# authors: Martin Ziqiao Ma (marstin@umich.edu),
#          Ben VanDerPloeg (bensvdp@umich.edu),
#          Owen Yidong Huang (owenhji@umich.edu),
#          Elina Eui-In Kim (euiink@umich.edu),
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import json
import random
from numpy.random import uniform

def config_gen(metaconfig_fname, output_fname):
    '''
    generate a config based on given metaconfig
    '''
    with open(metaconfig_fname) as f:
        metaconfig = json.load(f)

    res = {}
    res['map'] = random.choice(metaconfig['maps'])
    res['departure'] = random.choice(metaconfig['departures'])
    res['vehicle'] = random.choice(metaconfig['vehicles'])
    res['street_names'] = random.sample(metaconfig['street_names'], len(metaconfig['street_names'])) if metaconfig['street_name_shuffle'] else metaconfig['street_names']
    res['npc_count'] = int(random.uniform(metaconfig['npc_count_min'], metaconfig['npc_count_max']))

    if 'weather' in metaconfig:
        resweather = {}
        for par, (pmin, pmax) in metaconfig['weather'].items():
            resweather[par] = uniform(pmin, pmax)
        res['weather'] = resweather

    assets = []

    asset_groups = metaconfig['asset_groups'][res['map']]
    for ag in asset_groups:
        if 'asset' in ag: #single asset
            if 'probability' not in ag or random.random() < ag['probability']:
                assets.append({'type': ag['asset'], 'location': ag['transform']['location'], 'rotation': ag['transform']['rotation']})
        elif 'assets' in ag: #multiple
            transforms = random.sample(ag['transforms'], len(ag['transforms']))
            if len(transforms) < len(ag['assets']): #not enough transforms
                print('Not enough transforms in asset group! Skipping')
                continue

            for i, aname in enumerate(ag['assets']):
                if 'probabilities' not in ag or random.random() < ag['probabilities'][i]:
                    tf = transforms.pop()
                    assets.append({'type': aname, 'location': tf['location'], 'rotation': tf['rotation']})

        else: print('Invalid asset group in metaconfig: %s' % str(ag))

    res['assets'] = assets
    with open(output_fname, 'w') as f:
        json.dump(res, f, indent=4)

def storyboard_gen(template_fname, output_fname, simconfig):
    '''
    generate a storyboard based on given metaconfig and storyboard template
    '''
    with open(template_fname) as f:
        tpl = json.load(f)

    with open('../common/asset_metadata.json') as f:
        asset_md = json.load(f)

    assets = []
    for a in simconfig['assets']:
        ctg = asset_md[a['type']]['category']
        assets.append((a['type'], ctg))
    print(assets)
    asset_options = []
    for vname, vtype in tpl['variables']:
        eligible_assets = []
        for i, (_, actg) in enumerate(assets):
            if actg.startswith(vtype):
                eligible_assets.append(i)
        if not eligible_assets:
            print('ERROR: no eligible assets for variable %s. Storyboard gen failed' % vname)
            return None

        random.shuffle(eligible_assets)
        asset_options.append(eligible_assets)

    def recurse(aopts, taken_assets):
        available = [x for x in aopts[0] if x not in taken_assets]
        if not available: return None
        if len(aopts) == 1: return [available[0]]

        for ast in available:
            res = recurse(aopts[1:], taken_assets | {ast})
            if res: return [ast] + res

        return None

    assignments = recurse(asset_options, set())

    if not assignments:
        print('Storyboard generation failed')
        return None

    varnames = [n for n, _ in tpl['variables']]

    dps = {}
    for dp, dpexpr in tpl['dependents']:
        sbvar, attrname = dpexpr.split('.')
        sbvar_idx = varnames.index(sbvar)
        attr_val = asset_md[assets[assignments[sbvar_idx]][0]]['attributes'][attrname]

        if isinstance(attr_val, list):
            dps[dp] = random.choice(attr_val)
        else:
            dps[dp] = attr_val

    story = tpl['story']
    var_to_text = {}
    for i, (vname, _) in enumerate(tpl['variables']):
        vtext = asset_md[assets[assignments[i]][0]]['name']
        var_to_text[vname] = vtext
        story = story.replace('$' + vname, vtext)

    for dp, val in dps.items():
        var_to_text[dp] = val
        story = story.replace('$' + dp, val)

    sb_goals = []
    for tpl_goal in tpl['subgoals']:
        sb_goal = {}
        sb_goal['type'] = tpl_goal['type']

        if 'destination' in tpl_goal:
            if tpl_goal['destination'].startswith('$'):
                sb_goal['destination'] = var_to_text[tpl_goal['destination'][1:]]
            else:
                sb_goal['destination'] = tpl_goal['destination']
        if 'change_destination' in tpl_goal:
            if tpl_goal['change_destination'].startswith('$'):
                sb_goal['change_destination'] = var_to_text[tpl_goal['change_destination'][1:]]
            else:
                sb_goal['change_destination'] = tpl_goal['change_destination']


        desc = tpl_goal['description']
        for sbvar, sbval in var_to_text.items():
            desc = desc.replace('$' + sbvar, sbval)
        sb_goal['description'] = desc
        change = tpl_goal['change']if 'change' in tpl_goal else ''
        for sbvar, sbval in var_to_text.items():
            change = change.replace('$' + sbvar, sbval)
        sb_goal['change']=change
        sb_goal['delete_after_subgoal'] = tpl_goal['delete_after_subgoal'] if 'delete_after_subgoal' in tpl_goal else False
        sb_goal['trigger'] = tpl_goal['trigger'] if 'trigger' in tpl_goal else False
        sb_goal['delete'] = tpl_goal['delete'] if 'delete' in tpl_goal else False
        sb_goal['after'] = tpl_goal['after'] if 'after' in tpl_goal else []

        sb_goals.append(sb_goal)

    sb = {}
    sb['template'] = tpl
    sb['assignments'] = {vname: assignments[i] for i, (vname, _) in enumerate(tpl['variables'])} # maps variable name to an asset index which corresponds to the ordering of asset instance in simconfig
    sb['story'] = story
    sb['subgoals'] = sb_goals

    sb['hidden_from_co_wizard'] = [var_to_text[v] for v in tpl['hidden_from_co_wizard']] if 'hidden_from_co_wizard' in tpl else []
    if output_fname:
        with open(output_fname, 'w') as f:
            json.dump(sb, f)

    return sb
