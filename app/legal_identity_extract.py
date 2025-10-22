import re
from datetime import datetime

class LegalIdentity:
    def __init__(self, text, id_number = None, full_name = None, dob = None, gender = None, nationality = None, place_of_origin = None, place_of_residence = None):
        self.text = text
        if id_number is not None:
            self.id_number = id_number
        else:
            self.id_number = self.extract_id_number()
        if full_name is not None:
            self.full_name = full_name
        else:
            self.full_name = self.extract_full_name()
        if dob is not None:
            self.dob = dob
        else:
            self.dob = self.extract_dob()
        if gender is not None:
            self.gender = gender
        else:
            self.gender = self.extract_gender()
        if nationality is not None:
            self.nationality = nationality
        else:
            self.nationality = self.extract_nationality()
        if place_of_origin is not None:
            self.place_of_origin = place_of_origin
        else:
            self.place_of_origin = self.extract_place_of_origin()
        if place_of_residence is not None:
            self.place_of_residence = place_of_residence
        else:
            self.place_of_residence = self.extract_place_of_residence()

    def __str__(self):
        return (
                str(self.id_number) + ";"
                + str(self.full_name) + ";"
                + str(self.place_of_origin) + ";"
                + str(self.place_of_residence)
                + str(self.gender) + ";"
                + str(self.dob) + ";"
                + str(self.nationality) + ";"
        )

    def extract_id_number(self):
        return extract_pattern(self.text, r'No\.[:]*\s*(\d+)')

    def extract_full_name(self):
        return extract_pattern(self.text, r'Full name[:]*\s*\n(.+)')

    def extract_dob(self):
        raw_date = extract_pattern(self.text,r'Date of bic[:]*\s*(\d{4}/\d{4})')
        if  raw_date is None:
            raw_date = extract_pattern(self.text,r'Date of birth[:]*\s*(\d{2}/\d{2}/\d{4})')

        if is_valid_date_format(raw_date):
            return raw_date

        if raw_date:
            try:
                dob_obj = datetime.strptime(raw_date, "%d%m/%Y")
                return dob_obj.strftime("%d/%m/%Y")
            except ValueError:
                return None
        return None

    def extract_gender(self):
        gender = extract_pattern(self.text,r'Sex[:]*\s*(\w+)', re.IGNORECASE)
        if gender.upper() in ['NAM', 'MALE']:
            return 'M'
        else:
            return 'F'

    def extract_nationality(self):
        return extract_pattern(self.text,r'Nationality[:]*\s*(.+)')

    def extract_place_of_origin(self):
        return extract_pattern(self.text,r'Place of origin[:]*\s*(.+)')

    def extract_place_of_residence(self):
        return extract_pattern(self.text,
            r'Place of residence[:]*\s*(.+?)\s*(?:Ngày sinh|Giới tinh|Quốc tịch|$)',
            re.DOTALL
        )

def extract_pattern(text, pattern, flags=0):
    match = re.search(pattern, text, flags)
    return match.group(1).strip().replace('\n', ' ') if match else None

def is_valid_date_format(date_string):
    try:
        datetime.strptime(date_string, "%d/%m/%Y")
        return True
    except ValueError:
        return False

