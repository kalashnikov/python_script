import xml.etree.ElementTree as ET

class Temple:
    def __init__(self, name, god, district, address, religion_type, law_type, group_type, contact, principal):
         self.name = name
         self.god = god
         self.district = district
         self.address = address
         self.religion_type = religion_type
         self.law_type = law_type
         self.group_type = group_type
         self.contact = contact
         self.principal = principal

    def to_dict(self):
        return {
 	    'name': self.name,
 	    'god': self.god,
 	    'district': self.district,
 	    'address': self.address,
 	    'religion_type': self.religion_type,
 	    'law_type': self.law_type,
 	    'group_type': self.group_type,
 	    'contact': self.contact,
 	    'principal': self.principal
        }

    def __str__(self):
        return '### {} | God: {}, District: {}, Address: {}, Type({}, {}, {}), Contact: {}, Principal: {} \n'.format(
		self.name, self.god, self.district, self.address, self.religion_type, 
		self.law_type, self.group_type, self.contact, self.principal)

# Main
if __name__ == '__main__':
   
    all_temple = []     
    tree = ET.parse('temple.xml')
    root = tree.getroot()
    
    for t in root:
        t = Temple(t.findtext('Name'),
                t.findtext('God'),
                t.findtext('District'),
                t.findtext('Address'),
                t.findtext('Type'),
                t.findtext('LawType'),
                t.findtext('GroupType'),
                t.findtext('Contact'),
                t.findtext('Principal'))
        all_temple.append(t)
        print(t)

    #for t in all_temple:
    #    print(t)
