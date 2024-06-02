import frappe 
from langchain_community.utilities import SQLDatabase

"""This is for various api connection. Planned to use it later"""

@frappe.whitelist(allow_guest=True)
def jekono():
    dt = frappe.get_list('Communication',
        filters={
            'status': 'Open'
        },
        fields=['creation', 'communication_date'],
        start=10,
        page_length=20,
        as_list=True


     )
    return dt


@frappe.whitelist(allow_guest=True)
def any():
    # URI for MariaDB connection
    mariadb_uri = "mariadb+pymysql://root:Cse01306135@localhost:3306/_1746c1c5436f7769"

    # Created an SQLDatabase instance for MariaDB
    db = SQLDatabase.from_uri(mariadb_uri)

    
    dt = db.run("SELECT content FROM tabCommunication WHERE sent_or_received ='Received' ORDER BY creation DESC LIMIT 1;")
    print(dt)
    return dt

import frappe
from frappe.utils.pdf import get_pdf

@frappe.whitelist(allow_guest=True)
def generate_invoice():
    cart = {
        'Samsung Galaxy S20': 10,
        'iPhone 13': 80
    }

    html = '<h1>Rapid Lodgments</h1>'

    # Add items to PDF HTML
    html += '<ol>'
    for item, qty in cart.items():
        html += f'<li>{item} - {qty}</li>'
    html += '</ol>'

    # Attaching PDF to response
    frappe.local.response.filename = 'invoice.pdf'
    frappe.local.response.filecontent = get_pdf(html)
    pd = frappe.local.response.type = 'pdf'
    return pd




