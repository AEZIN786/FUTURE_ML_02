# Smart Support Ticket Classification & Priority Prediction

## 📖 Project Overview
This project implements an intelligent Machine Learning and NLP system designed to automate the processing of customer support tickets. By analyzing the text of incoming tickets, the system automatically classifies them into specific categories (e.g., Billing, Technical Issue, Account) and assigns a priority level (High, Medium, Low). This simulates a real-world SaaS customer support workflow, helping businesses respond faster and reduce manual sorting efforts.

## 💼 Business Problem
Customer support teams receive thousands of tickets daily. The biggest challenges include:
* Tickets are not categorized properly.
* Urgent issues get delayed in the queue.
* Support agents waste time sorting tickets instead of solving them.
This project provides a decision-support system that automatically routes and prioritizes tickets, improving operational efficiency and customer satisfaction.

## 📊 Dataset
The project uses a synthetic dataset of 10,000 realistic support tickets. 
* **Source:** Generated using `dataset/generate_data.py` to simulate real-world scenarios.
* **Features:** `ticket_text`, `category`, `priority`.
* **Classes:** 8 Categories (Billing, Technical Issue, Account, Refund, Payment, Shipping, Product Issue, General Inquiry) and 3 Priorities (High, Medium, Low).

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Support-Ticket-Classification.git
   cd Support-Ticket-Classification