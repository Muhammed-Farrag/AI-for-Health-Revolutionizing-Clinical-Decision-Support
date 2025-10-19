import xml.etree.ElementTree as ET
import re
import json
from typing import Dict, List
import os

class DrugInteractionExtractor:
    def __init__(self):
        self.namespace = {'ns0': 'http://www.drugbank.ca'}

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        text = re.sub(r'\(PubMed:\d+\)', '', text)
        text = re.sub(r'\(By similarity\)', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'<.*?>', '', text)

      
        text = re.sub(r'\s*\.+', '.', text)
        text = re.sub(r'[^a-zA-Z0-9.,;:%/+\-()\n\s]', '', text)  
        text = re.sub(r'\n+', ' ', text)          
        text = re.sub(r'\s+', ' ', text).strip()  
        text = text.lower()
        return text

    def extract_interactions(self, drug_elem) -> List[Dict]:
        """Extract known drug-drug interactions from a DrugBank XML element"""
        interactions = []
        seen = set()
        
        for interaction in drug_elem.findall('.//{*}drug-interaction'):
            name_elem = interaction.find('{*}name')
            desc_elem = interaction.find('{*}description')
            
            if name_elem is not None and name_elem.text:
                drug_name = self.clean_text(name_elem.text)
                if drug_name not in seen:
                    seen.add(drug_name)
                    
                    desc_text = desc_elem.text if desc_elem is not None else ""
                    interactions.append({
                        "drug_name": drug_name,
                        "description": self.clean_text(desc_text)
                    })
        return interactions

    def extract_food_interactions(self, drug_elem) -> List[str]:
        """Extract unique and cleaned food interactions"""
        foods = []
        seen = set()
        
        for food in drug_elem.findall('.//{*}food-interaction'):
            if food is not None and food.text:
                text = self.clean_text(food.text)
                if text and text not in seen:
                    seen.add(text)
                    foods.append(text)
        
        return foods
        


    def extract_synonyms(self, drug_elem) -> List[str]:
        """Extract unique drug synonyms"""
        synonyms = []
        seen = set()
        for syn in drug_elem.findall('.//{*}synonym'):
            if syn is not None and syn.text:
                clean_syn = self.clean_text(syn.text)
                if clean_syn and clean_syn not in seen:
                    seen.add(clean_syn)
                    synonyms.append(clean_syn)
        return synonyms


    def extract_drug_data(self, drug_elem) -> Dict:
        """Extract only required fields from a DrugBank drug element"""
        name_elem = drug_elem.find('{*}name')
        desc_elem = drug_elem.find('{*}description')
        
        name_text = name_elem.text if name_elem is not None else ""
        desc_text = desc_elem.text if desc_elem is not None else ""

        drug_data = {
            "name": self.clean_text(name_text),
            "synonyms": self.extract_synonyms(drug_elem)[:5],
            "description": self.clean_text(desc_text),
            "known_interactions": self.extract_interactions(drug_elem),
            "food_interactions": self.extract_food_interactions(drug_elem)
        }

        return {k: v for k, v in drug_data.items() if v}

    def process_xml(self, xml_file: str, output_file: str = 'ddi_drugs.json'):
        """Process XML and save simplified JSON — namespace-independent version"""
        import xml.etree.ElementTree as ET
        import json

        tree = ET.parse(xml_file)
        root = tree.getroot()

        drugs = []
        seen_ids = set()

        for drug in root.findall('.//{*}drug'):
            drugbank_id_elem = drug.find('.//{*}drugbank-id[@primary="true"]')
            drugbank_id = self.clean_text(drugbank_id_elem.text) if drugbank_id_elem is not None else None

            if not drugbank_id or drugbank_id in seen_ids:
                continue
            seen_ids.add(drugbank_id)

            drug_data = self.extract_drug_data(drug)
            drug_data['drugbank_id'] = drugbank_id  
            drugs.append(drug_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(drugs, f, indent=2, ensure_ascii=False)

        print(f"Processed {len(drugs)} drugs and saved to {output_file}")
        return drugs




if __name__ == "__main__":
    extractor = DrugInteractionExtractor()

    input_folder = "db"
    output_folder = "cleaned_db"

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".xml"):
            xml_path = os.path.join(input_folder, filename)
            json_path = os.path.join(output_folder, filename.replace(".xml", ".json"))

            drugs = extractor.process_xml(xml_path, json_path)

           
            if drugs:
                print("\n Sample from cleaned database", filename)
                print(json.dumps(drugs[0], indent=2, ensure_ascii=False))
