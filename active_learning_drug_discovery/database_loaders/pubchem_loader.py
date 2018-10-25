"""
    Contains functions to load and standardize PubChem data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import xml.etree.ElementTree as ET
import requests
import urllib
import time
import os

# PubChem Globals
PC_CGI = 'https://pubchem.ncbi.nlm.nih.gov/pug/pug.cgi'

def load_assay(aid_list, xml_req_file, xml_poll_file, output_dir):
    """Loads PubChem assay data.
        # Arguments
            aid_list: list of assay ids.
            xml_req_file: base xml file for PubChem requests. See https://pubchemdocs.ncbi.nlm.nih.gov/power-user-gateway.
            xml_poll_file: base xml file for PubChem polling. See https://pubchemdocs.ncbi.nlm.nih.gov/power-user-gateway.
            output_dir: specifies directory to store assay data. 
        # Returns
            True if downlownd successful. 
        # Raises
            ValueError: In case POST requests fail.
    """ 
    if not isinstance(aid_list, list):
        aid_list = [aid_list]
    
    # submit xml post to PC_CGI
    print('Submitting request to PubChem...')
    req_tree = ET.parse(xml_req_file)
    req_root = req_tree.getroot()
    subroot = list(req_root.iter('PCT-ID-List_uids'))
    if len(subroot) == 0:
        raise ValueError('No <PCT-ID-List_uids> element found.'
                         ' Check if %s has correct format.' % xml_req_file)
    subroot = subroot[0]
    for aid in aid_list:                     
        aid_element = ET.Element('PCT-ID-List_uids_E')
        aid_element.text = str(aid)
        aid_element.tail = '\n'
        subroot.append(aid_element)

    xml_post_data = ET.tostring(req_root)
    req_reply = requests.post(PC_CGI, data=xml_post_data)
    # process reply, extract info, and poll until done
    print('Polling request...\r')
    req_status = 'running'
    poll_reply = req_reply
    while req_status == 'running':
        time.sleep(4)
        rr_tree = ET.ElementTree(ET.fromstring(poll_reply.text))
        rr_root = rr_tree.getroot()
        req_status = list(rr_root.iter('PCT-Status'))[0].attrib['value']
        
        if req_status == 'running':
            req_id = list(rr_root.iter('PCT-Waiting_reqid'))[0].text

            poll_tree = ET.parse(xml_poll_file)
            poll_root = poll_tree.getroot()
            req_id_element = list(poll_root.iter('PCT-Request_reqid'))
            if len(req_id_element) == 0:
                raise ValueError('No <PCT-Request_reqid> element found.'
                                 ' Check if %s has correct format.' % xml_poll_file)
            req_id_element = req_id_element[0]
            req_id_element.text = req_id
            xml_poll_data = ET.tostring(poll_root)
            poll_reply = requests.post(PC_CGI, data=xml_poll_data)
        
    if req_status != 'success':
        raise ValueError('Non-successful result retrieved. Check xml files.'
                         ' Request status: %s.' % req_status)

    # write to output directory
    print('Downloading file...\r')
    pr_tree = ET.ElementTree(ET.fromstring(poll_reply.text))
    pr_root = pr_tree.getroot()
    download_url = list(pr_root.iter('PCT-Download-URL_url'))[0].text
    filename = output_dir+'/aid_{}.csv.gz'.format('_'.join(aid_list))
    download_res = urllib.request.urlretrieve(download_url, filename)
    print('Download complete.')
    
    return os.path.exists(filename)
    
if __name__ == '__main__':
    output_dir = './'
    xml_req_file = './pubchem_assayreq.xml'
    xml_poll_file = './pubchem_assaypoll.xml'
    aid_list = ['523', '820']

    assert load_assay(aid_list, xml_req_file, xml_poll_file, output_dir)