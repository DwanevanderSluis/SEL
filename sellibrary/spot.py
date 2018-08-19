import logging


class Spot:
    def __init__(self, entity_id, start_char, end_char, text):
        # set up instance variables
        self.entity_id = entity_id
        self.start_char = start_char
        self.end_char = end_char
        self.text = text

    def __repr__(self):
        return "<Spot text:%s entity_id:%d start_char:%d end_char:%d  >" % (self.text, self.entity_id, self.start_char, self.end_char  )

    def __str__(self):
        return "From str method of Spot: text:%s entity_id:%d start_char:%d end_char:%d " % (self.text, self.entity_id, self.start_char, self.end_char  )


    def get_entity_id(self):
        return self.entity_id

    def get_start_char(self):
        return self.start_char
