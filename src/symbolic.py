import pysdd
from pysdd.sdd import Vtree, SddManager, WmcManager
from pysdd.iterator import SddIterator
from array import array
from graphviz import Source
import sys, os


from pathlib import Path
here = Path(__file__).parent


# input: [a, b, c] 
# output: (a & ~b & ~c) | (~a & b & ~c) | (~a & ~b & c) | (~a & ~b & ~c)
def AtMostOne(lits, mgr):
	alpha = mgr.false()		
	for lit in lits:
		beta = alpha | ~lit

	alpha = beta | exactly_one(lits, mgr)        
	return alpha

# def implication(literals, mgr):

# 	alpha = mgr.false()

# 	beta0 = literals[0]
# 	for lit in literals[1:]:
# 		beta = ~lit | beta0  			
# 		alpha = alpha | beta
# 	return alpha


# input: [a, b, c] 
# output: (a & ~b & ~c) | (~a & b & ~c) | (~a & ~b & c)
def exactly_one(lits, mgr):  
	alpha = mgr.false()
	for lit in lits:
		beta = lit
		for lit2 in lits:
			if lit2!=lit:
				beta = beta & ~lit2
		alpha = alpha | beta
	return alpha

  
def main(*args):


    """
	Creates the vtree and sdd format files of each dataset. 
	"""
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--label_size', type=int)
    args = parser.parse_args()

    dataset_name = args.dataset
    label_size = args.label_size
    # set up vtree and manager
    vtree = Vtree(label_size)
    sdd_mgr = SddManager(vtree = vtree)	
    folder ='{}/{}/{}/'.format(here, "sdd_input", dataset_name)
    if not os.path.exists(os.path.dirname(folder)):
        try:
            os.makedirs(os.path.dirname(folder))
        except OSError as exc:
            if exc.errno != errno.EEXITST:
                raise

    if args.dataset == 'bbn_modified':
        ANIMAL,\
        CONTACT_INFO,ADDRESS,PHONE,\
        DISEASE,\
        EVENT,HURRICANE,WAR, \
        FAC,AIRPORT,ATTRACTION,BRIDGE,BUILDING,HIGHWAY_STREET,\
        GAME,\
        GPE,CITY,COUNTRY,STATE_PROVINCE,\
        LANGUAGE,\
        LAW,\
        LOCATION,CONTINENT,LAKE_SEA_OCEAN,REGION,RIVER,\
        ORGANIZATION,CORPORATION,EDUCATIONAL,GOVERNMENT,HOSPITAL,HOTEL,POLITICAL,RELIGIOUS,\
        PERSON,\
        PLANT,\
        PRODUCT,VEHICLE,WEAPON,\
        SUBSTANCE,CHEMICAL,DRUG,FOOD,\
        WORK_OF_ART,BOOK,PAINTING,PLAY,SONG = sdd_mgr.vars

        implication1 = CONTACT_INFO | (~ADDRESS & ~PHONE)
        implication2 = EVENT | (~HURRICANE  & ~WAR)
        implication3 = FAC | (~AIRPORT  & ~ATTRACTION  & ~BRIDGE  & ~BUILDING  & ~HIGHWAY_STREET)
        implication4 = GPE | (~CITY  & ~COUNTRY  & ~STATE_PROVINCE)
        implication5 = LOCATION  | (~CONTINENT & ~LAKE_SEA_OCEAN  & ~REGION  & ~RIVER )
        implication6 = ORGANIZATION | (~CORPORATION  & ~EDUCATIONAL  & ~GOVERNMENT  & ~HOSPITAL  & ~HOTEL & ~POLITICAL & ~RELIGIOUS )
        implication7 = PRODUCT  | (~VEHICLE  & ~WEAPON)
        implication8 = SUBSTANCE | (~CHEMICAL  & ~DRUG  & ~FOOD)
        implication9 = WORK_OF_ART | (~BOOK & ~PAINTING  & ~PLAY  & ~SONG)

        at_most_one1 = AtMostOne((ADDRESS,PHONE), sdd_mgr)
        at_most_one2 = AtMostOne((HURRICANE,WAR), sdd_mgr)
        at_most_one3 = AtMostOne((AIRPORT,ATTRACTION,BRIDGE,BUILDING,HIGHWAY_STREET), sdd_mgr)
        at_most_one4 = AtMostOne((CITY,COUNTRY,STATE_PROVINCE), sdd_mgr)
        at_most_one5 = AtMostOne((CONTINENT,LAKE_SEA_OCEAN,REGION,RIVER), sdd_mgr)
        at_most_one6 = AtMostOne((CORPORATION,EDUCATIONAL,GOVERNMENT,HOSPITAL,HOTEL,POLITICAL,RELIGIOUS), sdd_mgr)
        at_most_one7 = AtMostOne((VEHICLE,WEAPON), sdd_mgr)
        at_most_one8 = AtMostOne((CHEMICAL,DRUG, FOOD), sdd_mgr)
        at_most_one9 = AtMostOne((BOOK,PAINTING, PLAY, SONG), sdd_mgr)

        mutually_exclusive = exactly_one((ANIMAL, CONTACT_INFO, DISEASE, EVENT, FAC, GAME, GPE, \
            LANGUAGE, LAW, LOCATION, ORGANIZATION, PERSON, PLANT, PRODUCT, SUBSTANCE, WORK_OF_ART), sdd_mgr)

        et_logical_formula = mutually_exclusive \
            & at_most_one9 & at_most_one8 & at_most_one7 & at_most_one6 & at_most_one5 \
            & at_most_one4 & at_most_one3 & at_most_one2 & at_most_one1 \
            & implication1 & implication2 & implication3 & implication4 \
            & implication5 & implication6 & implication7 & implication8 & implication9

    elif args.dataset == 'ontonotes_modified':	
        # label_size = 72
        location,celestial,city,country,geography,body_of_water,park,structure,airport,government,hotel,sports_facility,\
            transit,bridge,railway,road,organization,company,broadcast,news,education,government,military,political_party,\
            sports_team,stock_exchange,other,art,broadcast,film,music,stage,writing,body_part,currency,event,election,\
            holiday,natural_disaster,protest,sports_event,violent_conflict,food,health,treatment,heritage,legal,living_thing,\
            animal,product,car,computer,software,weapon,religion,scientific,sports_and_leisure,person,artist,actor,author,music,\
            athlete,business,doctor,education,student,teacher,legal,military,political_figure,title  = sdd_mgr.vars
        implication1 = location | (~celestial & ~city& ~country & ~geography & ~park & ~structure & ~transit)
        implication2 = geography | ~body_of_water 
        implication3 = structure | (~airport & ~government & ~hotel & ~sports_facility)
        implication4 = transit| (~bridge  & ~railway  & ~road)
        implication5 = organization | (~company & ~education  & ~government  & ~military  & ~political_party & ~sports_team & ~stock_exchange)
        implication6 = company  | (~broadcast & ~news)
        implication7 = other | ( ~art & ~body_part & ~currency & ~event & ~food & ~health & ~heritage & ~legal & ~living_thing & ~product & ~religion & ~scientific & ~sports_and_leisure)
        implication8 = art | (~broadcast & ~film & ~music & ~stage & ~writing)
        implication9 = event | (~election & ~holiday  & ~natural_disaster  &~protest & ~sports_event & ~violent_conflict)
        implication10 = health | ~treatment
        implication11 = living_thing | ~animal
        implication12 = product | (~car & ~computer & ~software & ~weapon)
        implication13 = person | (~artist & ~athlete & ~business & ~doctor & ~education & ~legal & ~military & ~political_figure & ~title)
        implication14 = artist | (~actor & ~author & ~music)
        implication15 = education | (~student & ~teacher)
        mutually_exclusive = exactly_one((location, other, organization, person),sdd_mgr)
        et_logical_formula = mutually_exclusive & (implication1 & implication2 & implication3 & implication4 )| (implication5 & implication6 ) | (implication7 & implication8 & implication9 & implication10 & implication11 &implication12) |( implication13 & implication14 & implication15 )
    
    elif args.dataset == 'ontonotes_original':
        location,celestial,city,country,geography,body_of_water,island,mountain,geograpy,island,park,structure,airport,government,hospital,hotel,restaurant,sports_facility,theater,transit,bridge,railway,road,organization,company,broadcast,news,education,government,military,music,political_party,sports_league,sports_team,stock_exchange,transit,other,art,broadcast,film,music,stage,writing,award,body_part,currency,event,accident,election,holiday,natural_disaster,protest,sports_event,violent_conflict,food,health,malady,treatment,heritage,internet,language,programming_language,legal,living_thing,animal,product,car,computer,mobile_phone,software,weapon,religion,scientific,sports_and_leisure,supernatural,person,artist,actor,author,director,music,athlete,business,coach,doctor,education,student,teacher,legal,military,political_figure,religious_leader,title = sdd_mgr.vars
        implication1 = location | (~celestial & ~ city& ~country & ~geography & ~park & ~structure & ~transit )
        implication2 = geography | (~body_of_water & ~island & ~mountain)
        implication3 = structure | (~airport & ~government & ~hospital & ~hotel & ~restaurant & ~sports_facility  & ~theater )
        implication4 = transit| (~bridge  & ~railway  & ~road)
        implication5 = organization | (~company & ~education  & ~government  & ~military & ~music & ~political_party  & ~sports_league & ~sports_team & ~stock_exchange & ~transit)
        implication6 = company  | (~broadcast & ~news)
        implication7 = other | ( ~art & ~award & ~body_part & ~currency & ~event & ~food & ~health & ~heritage & ~internet & ~language & ~legal & ~living_thing & ~product & ~religion & ~scientific & ~sports_and_leisure & ~supernatural)
        implication8 = art | (~broadcast & ~film & ~music & ~stage & ~writing)
        implication9 = event | (~accident & ~election & ~holiday  & ~natural_disaster  &~protest & ~sports_event & ~violent_conflict)
        implication10 = health | ( ~malady & ~treatment)
        implication11 = language | ~programming_language
        implication12 = living_thing | ~animal
        implication13 = product | (~car & ~computer & ~software & ~weapon & ~mobile_phone)
        implication14 = person | (~artist & ~athlete & ~business & ~coach & ~doctor & ~education & ~legal & ~military & ~political_figure & ~religious_leader & ~title)
        implication15 = artist | (~actor & ~author & ~director & ~music)
        implication16 = education | (~student & ~teacher)
        et_logical_formula = (implication1 & implication2 & implication3 & implication4 )| (implication5 & implication6 ) | (implication7 & implication8 & implication9 & implication10 & implication11 &implication12 & implication13) |( implication14 & implication15 & implication16 )
        

    print("saving sdd and vtree ... ")
    with open(folder+"sdd.dot", "w") as out:
        print(et_logical_formula.dot(), file = out)

    with open(folder +"vtree.dot", "w") as out:
        print(vtree.dot(), file=out)

    wmc_mgr = et_logical_formula.wmc(log_mode = True)
    sdd_nodes = wmc_mgr.node

    vtree.save(bytes(here/folder/"et.vtree"))
   
    print(bytes(here/folder/"et.vtree"))
    sdd_mgr.save(bytes(here/folder/"et.sdd"), sdd_nodes)
    print("done")



if __name__ == "__main__":
    main()
