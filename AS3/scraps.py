import random

from AS2.chromosome import chromosome


# A box or bin that holds some items and has an id so we can track it.
class Box:
    def __init__(self, box_id_in, item):
        # create a new box assign the id and an item.
        self.box_id = box_id_in
        self.items = []
        self.add_item(item)

    def __repr__(self):
        return f"Box(box_id={self.box_id}, total_weight={self.get_total_weight()}) \n"

    def add_item(self, item):
        item.box_id = self.box_id
        self.items.append(item)

    def get_total_weight(self):
        return round(sum(item.weight for item in self.items), 1)

 #   if offspring1.bins.get(item1.box_id):
        # if the box id already existed, add the items.
 #       offspring1.bins[item1.box_id].add_item(copy.deepcopy(item1))
 #   else:
        # otherwise add the box.
#        offspring1.bins[item1.box_id] = Box(item1.box_id, item1)
#    if offspring2.bins.get(item2.box_id):
        # if the box id already existed, add the items.
#        offspring2.bins[item2.box_id].add_item(copy.deepcopy(item2))
#   else:
        # otherwise add the box.
 #       offspring2.bins[item2.box_id] = Box(item2.box_id, item1)

## assign items to bins efficiently and under weight limits.
## this one is not random :/
def assign_items_to_bins(self, item_weights):
    bins = {}  # bin_id -> total weight
    # loop through the items and put them in a box.
    # worst case we will get 1 item per box
    for weight in item_weights:
        # Try assigning to an existing bin first
        assigned = False
        bin_ids = list(bins.keys())
        random.shuffle(bin_ids)

        for bin_id in bin_ids:
            # if we have not reached max weight for a box after adding the next item...
            if bins[bin_id].get_total_weight() + weight <= MAX_BOX_WEIGHT:
                # Add the box id assigned to the list of bin assignments.
                self.bin_assignments.append(bin_id)
                # throw the item in the box!
                bins[bin_id].items.append(weight)
                # set our flag so we don't open a new box.
                assigned = True
                break

        # If not assignable to any existing box, create a new one
        if not assigned:
            new_bin_id = len(bins)
            # Add the box id assigned to the list of bin assignments.
            self.bin_assignments.append(new_bin_id)
            # grab a new box, give it a new id and throw the item in.
            bins[new_bin_id] = Box(new_bin_id, weight)



            ############################3
            # FITNESS ATTEMPTS
            ###############################3

            # instead use the max weight in the fewest boxes. (which also goes to zero very quickly)
            # normalized_efficiency = len(self.bin_assignments) / self.get_total_weight()
            # return normalized_efficiency

            # fitness that takes min number of boxes with max weight and max item count.
            total_items = sum(len(box.item_weights) for box in self.bins.values())
            num_bins_used = self.get_number_of_bins_used()
            total_weight = self.get_total_weight()
            max_weight = self.num_items * chromosome.MAX_ITEM_WEIGHT

            # Normalize components
            item_score = total_items / self.num_items  # 0 to 1
            weight_score = total_weight / max_weight  # 0 to 1
            box_penalty = self.get_number_of_bins_used() / self.num_items  # smaller is better

            # Combine: maximize items and weight, minimize boxes
            fitness = (2 * item_score + 2 * weight_score) - box_penalty
            print("fitness:", fitness)
            return fitness