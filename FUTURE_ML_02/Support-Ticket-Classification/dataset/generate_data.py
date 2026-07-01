import pandas as pd
import numpy as np
import os
import random
from typing import Dict, List

def generate_dataset(num_samples: int = 10000, output_path: str = 'dataset/support_tickets.csv') -> None:
    """
    Generates a realistic synthetic dataset of support tickets.
    
    Args:
        num_samples (int): Number of tickets to generate.
        output_path (str): Path to save the generated CSV file.
    """
    categories = [
        'Billing', 'Technical Issue', 'Account', 'Refund', 
        'Payment', 'Shipping', 'Product Issue', 'General Inquiry'
    ]
    priorities = ['High', 'Medium', 'Low']
    
    templates: Dict[str, Dict[str, List[str]]] = {
        'Billing': {
            'High': [
                "I was charged twice for my subscription and my account is now overdrawn. Please fix this immediately!",
                "My credit card was charged but the service is not active. I need a refund right now.",
                "There is an unauthorized charge on my bill. I need this resolved ASAP."
            ],
            'Medium': [
                "I have a question about my latest invoice. Can someone explain the charges?",
                "My billing address needs to be updated for the next cycle.",
                "I didn't receive my invoice for this month."
            ],
            'Low': [
                "Just wondering when the next billing cycle starts.",
                "Is there a way to get a copy of my old invoices?",
                "How do I change my payment method for future bills?"
            ]
        },
        'Technical Issue': {
            'High': [
                "The entire system is down! None of my team can log in. This is critical!",
                "I'm getting a 500 Internal Server Error every time I try to save my work. I'm losing data!",
                "The application crashes immediately upon opening. I cannot work at all."
            ],
            'Medium': [
                "The dashboard is loading very slowly today.",
                "I can't export my reports to PDF. It just hangs.",
                "The search function is not returning any results."
            ],
            'Low': [
                "The font on the settings page looks a bit weird.",
                "Is there a dark mode available for the app?",
                "Minor UI glitch on the profile page."
            ]
        },
        'Account': {
            'High': [
                "My account has been hacked! Someone changed my password and email. Lock it immediately!",
                "I cannot log in at all. I've tried resetting my password 5 times and it's not working.",
                "My account was suspended without any notice. I need access restored now."
            ],
            'Medium': [
                "I need to merge two duplicate accounts.",
                "How do I change my email address associated with this account?",
                "I want to delete my account. Please guide me through the process."
            ],
            'Low': [
                "How do I update my profile picture?",
                "Can I change my username?",
                "Where can I find my account ID?"
            ]
        },
        'Refund': {
            'High': [
                "I was promised a refund 3 weeks ago and still haven't received it. This is unacceptable!",
                "The product was completely broken and I need my money back immediately.",
                "I cancelled my subscription within the 30-day window but was still charged. Refund now!"
            ],
            'Medium': [
                "I would like to request a partial refund for the damaged item.",
                "When can I expect my refund to be processed?",
                "I returned the item last week, when will the refund hit my account?"
            ],
            'Low': [
                "What is your refund policy?",
                "Do you offer refunds for digital products?",
                "Just checking the status of my refund request."
            ]
        },
        'Payment': {
            'High': [
                "My payment failed but the money was deducted from my bank! Fix this now!",
                "I cannot complete the checkout process. The payment gateway is timing out.",
                "My card is being declined even though it has sufficient funds. I need to buy this now."
            ],
            'Medium': [
                "How do I add a new credit card to my account?",
                "My payment is pending. How long does it usually take to clear?",
                "Can I pay via PayPal instead of credit card?"
            ],
            'Low': [
                "Do you accept cryptocurrency?",
                "What currencies do you support for payments?",
                "Is there a fee for using a specific payment method?"
            ]
        },
        'Shipping': {
            'High': [
                "My package was marked as delivered but I never received it! I need this resolved immediately.",
                "The shipping address was wrong and it's going to the wrong state. Stop the shipment!",
                "My order is 2 weeks late and I need the items for an event tomorrow."
            ],
            'Medium': [
                "Can I change the shipping address for my current order?",
                "How much is the expedited shipping fee?",
                "My tracking number hasn't updated in 3 days."
            ],
            'Low': [
                "Do you ship internationally?",
                "What courier do you use for standard shipping?",
                "When will my order be shipped?"
            ]
        },
        'Product Issue': {
            'High': [
                "The product caught fire! This is a massive safety hazard. I need a replacement and an investigation!",
                "The device stopped working completely after one day. It's totally defective.",
                "The software is corrupting my files. I need a patch immediately."
            ],
            'Medium': [
                "The battery life on this device is much shorter than advertised.",
                "The color of the product I received doesn't match the website.",
                "One of the features is not working as described in the manual."
            ],
            'Low': [
                "The packaging was a bit dented, but the product is fine.",
                "The instructions were a bit hard to understand.",
                "Is there a way to customize the color of the app interface?"
            ]
        },
        'General Inquiry': {
            'High': [
                "I need to speak to a manager immediately regarding a legal matter.",
                "Your system is violating GDPR and I need an urgent explanation.",
                "I am a press member and need an immediate statement on the recent outage."
            ],
            'Medium': [
                "Do you offer discounts for non-profit organizations?",
                "Can I get a demo of the enterprise features?",
                "How do I become an affiliate partner?"
            ],
            'Low': [
                "What are your office hours?",
                "Do you have a mobile app?",
                "Where can I read your blog?"
            ]
        }
    }
    
    prefixes = ["Hello, ", "Hi support, ", "Urgent: ", "Help! ", ""]
    suffixes = [" Thanks.", " Please help.", " Waiting for your reply.", " ASAP.", ""]
    
    data = []
    for _ in range(num_samples):
        category = random.choice(categories)
        priority = random.choice(priorities)
        
        base_text = random.choice(templates[category][priority])
        text = random.choice(prefixes) + base_text + random.choice(suffixes)
        
        data.append({
            'ticket_text': text,
            'category': category,
            'priority': priority
        })
        
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at {output_path} with {num_samples} samples.")

if __name__ == "__main__":
    generate_dataset(10000)