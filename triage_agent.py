#!/usr/bin/env python3
"""
Multi-Domain Support Triage Agent
Handles HackerRank, Claude, and Visa support tickets with safety-first escalation
"""

import pandas as pd
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import argparse
from datetime import datetime

@dataclass
class TriageDecision:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str

class RequestType(Enum):
    PRODUCT_ISSUE = "product_issue"
    FEATURE_REQUEST = "feature_request"
    BUG = "bug"
    INVALID = "invalid"
    BILLING = "billing"
    SECURITY = "security"
    ACCOUNT = "account"

class CompanyDomain(Enum):
    HACKERRANK = "HackerRank"
    CLAUDE = "Claude"
    VISA = "Visa"
    NONE = "None"

class SupportCorpus:
    """Knowledge base derived from support documentation patterns"""
    
    def __init__(self):
        self.corpus = {
            # HackerRank domains
            "HackerRank": {
                "product_areas": [
                    "Account Management", "Assessments", "Interview", 
                    "Billing & Payments", "API", "Company Admin",
                    "Technical Issues", "Proctoring"
                ],
                "high_risk_keywords": [
                    "password", "login", "credentials", "security", 
                    "fraud", "payment failed", "refund", "delete account"
                ],
                "faq_patterns": [
                    r"how to.*create.*account",
                    r"reset.*password",
                    r"cannot.*login",
                    r"assessment.*not.*loading",
                    r"interview.*link.*not.*working"
                ]
            },
            # Claude domains  
            "Claude": {
                "product_areas": [
                    "Account & Billing", "Usage Limits", "API Access",
                    "Projects", "Conversations", "Mobile App",
                    "Privacy & Security"
                ],
                "high_risk_keywords": [
                    "billing", "subscription", "cancel", "refund",
                    "delete data", "privacy", "export conversations"
                ],
                "faq_patterns": [
                    r"usage.*limit",
                    r"api.*key",
                    r"project.*not.*loading",
                    r"message.*history",
                    r"mobile.*app.*crash"
                ]
            },
            # Visa domains
            "Visa": {
                "product_areas": [
                    "Card Services", "Online Payments", "Disputes & Claims",
                    "Security & Fraud", "Merchant Support", "Contactless"
                ],
                "high_risk_keywords": [
                    "fraud", "dispute", "chargeback", "lost card",
                    "stolen", "unauthorized", "block card"
                ],
                "faq_patterns": [
                    r"transaction.*declined",
                    r"contactless.*not.*working",
                    r"merchant.*dispute",
                    r"pin.*forgot",
                    r"card.*blocked"
                ]
            }
        }
    
    def get_domain(self, company: str) -> Dict:
        company = company.title()
        for domain, data in self.corpus.items():
            if domain in company or company in domain:
                return data
        return self.corpus["HackerRank"]  # default fallback

class TriageAgent:
    def __init__(self):
        self.corpus = SupportCorpus()
        self.risk_score_threshold = 0.7
        self.escalation_keywords = {
            "security", "fraud", "legal", "lawsuit", "credentials",
            "password", "ssn", "financial", "dispute", "refund urgent"
        }
    
    def analyze_ticket(self, issue: str, subject: str, company: str) -> TriageDecision:
        """Main triage logic"""
        full_text = f"{subject} {issue}".lower()
        
        # Determine company domain
        if company == "None" or pd.isna(company):
            company_domain = self._infer_company(full_text)
        else:
            company_domain = company
        
        domain_data = self.corpus.get_domain(company_domain)
        
        # Classify request type
        request_type = self._classify_request_type(full_text)
        
        # Assess risk and urgency
        risk_score = self._calculate_risk_score(full_text, domain_data)
        urgency = self._assess_urgency(full_text)
        
        # Determine product area
        product_area = self._identify_product_area(full_text, company_domain, domain_data)
        
        # Decide status
        status, justification = self._decide_status(
            full_text, risk_score, urgency, request_type, company_domain
        )
        
        # Generate response
        response = self._generate_response(
            status, full_text, product_area, company_domain, domain_data
        )
        
        return TriageDecision(
            status=status,
            product_area=product_area,
            response=response,
            justification=justification,
            request_type=request_type.value if isinstance(request_type, RequestType) else request_type
        )
    
    def _infer_company(self, text: str) -> str:
        """Infer company from content"""
        hacker_indicators = ["hackerrank", "assessment", "interview", "coding test"]
        claude_indicators = ["claude", "anthropic", "conversation", "project"]
        visa_indicators = ["visa", "card", "payment", "transaction", "merchant"]
        
        if any(word in text for word in hacker_indicators):
            return "HackerRank"
        elif any(word in text for word in claude_indicators):
            return "Claude"
        elif any(word in text for word in visa_indicators):
            return "Visa"
        return "HackerRank"  # default
    
    def _classify_request_type(self, text: str) -> str:
        """Classify the type of request"""
        bug_indicators = ["bug", "error", "crash", "broken", "not working", "fails"]
        feature_indicators = ["feature", "request", "add", "new", "support for"]
        billing_indicators = ["billing", "payment", "subscription", "invoice"]
        security_indicators = ["security", "fraud", "password", "login"]
        
        if any(ind in text for ind in security_indicators):
            return "security"
        elif any(ind in text for ind in billing_indicators):
            return "billing"
        elif any(ind in text for ind in bug_indicators):
            return "bug"
        elif any(ind in text for ind in feature_indicators):
            return "feature_request"
        elif "how to" in text or "what is" in text:
            return "product_issue"
        else:
            return "invalid"
    
    def _calculate_risk_score(self, text: str, domain_data: Dict) -> float:
        """Calculate risk score (0-1)"""
        score = 0.0
        word_count = len(text.split())
        
        # High risk keywords
        for keyword in self.escalation_keywords | set(domain_data["high_risk_keywords"]):
            if keyword in text:
                score += 0.3
        
        # Urgency indicators
        urgency_words = ["urgent", "emergency", "immediately", "asap", "critical"]
        for word in urgency_words:
            if word in text:
                score += 0.2
        
        # Financial/legal sensitivity
        sensitive_patterns = ["money", "refund", "chargeback", "lawsuit", "legal"]
        for pattern in sensitive_patterns:
            if pattern in text:
                score += 0.25
        
        return min(score, 1.0)
    
    def _assess_urgency(self, text: str) -> str:
        """Assess urgency level"""
        urgent_indicators = ["urgent", "emergency", "asap", "immediately", "critical", "now"]
        if any(ind in text for ind in urgent_indicators):
            return "high"
        return "normal"
    
    def _identify_product_area(self, text: str, company: str, domain_data: Dict) -> str:
        """Identify most relevant product area"""
        areas = domain_data["product_areas"]
        scores = {}
        
        for area in areas:
            score = sum(1 for word in area.lower().split() if word in text)
            scores[area] = score
        
        if scores:
            return max(scores, key=scores.get)
        return areas[0]  # fallback
    
    def _decide_status(self, text: str, risk_score: float, urgency: str, 
                      request_type: str, company: str) -> Tuple[str, str]:
        """Decide whether to reply or escalate"""
        
        # Always escalate high-risk cases
        if risk_score > self.risk_score_threshold:
            return "escalated", f"High risk score ({risk_score:.2f}) - {request_type} requires human review"
        
        # Escalate security/billing/fraud
        if request_type in ["security", "billing"]:
            return "escalated", f"Sensitive {request_type} issue requires specialist handling"
        
        # Escalate invalid/malicious content
        malicious_indicators = ["hack", "exploit", "bypass", "crack", "illegal"]
        if any(ind in text for ind in malicious_indicators):
            return "escalated", "Potential malicious intent detected"
        
        # Safe to reply for FAQs and simple issues
        return "replied", f"Standard {request_type} - safe to handle automatically (risk: {risk_score:.2f})"
    
    def _generate_response(self, status: str, text: str, product_area: str, 
                          company: str, domain_data: Dict) -> str:
        """Generate appropriate response"""
        if status == "escalated":
            return f"""Thank you for contacting {company} support. 

Your issue in {product_area} has been flagged for priority review by our specialist team and will be addressed within 24 hours.

Reference: {datetime.now().strftime('%Y%m%d-%H%M%S')}
Ticket Status: Escalated for human review"""

        # FAQ-style responses
        faq_responses = {
            "account": "For account issues, please visit your account settings or use the password recovery flow.",
            "assessment": "Assessment issues are typically resolved by refreshing the page or checking your internet connection.",
            "payment": "Payment issues require manual review. Our billing team will contact you shortly.",
            "api": "API documentation is available at our developer portal."
        }
        
        # Match to best FAQ
        for key, response in faq_responses.items():
            if key in text:
                return f"""Hi there,

{response}

If this doesn't resolve your issue, please provide more details.

Thanks,
{company} Support Team"""
        
        return f"""Thank you for your message regarding {product_area}.

We're reviewing your request and will follow up shortly with specific guidance.

Best,
{company} Support"""

def main():
    parser = argparse.ArgumentParser(description="Multi-Domain Support Triage Agent")
    parser.add_argument("input_file", help="Path to support_tickets.csv")
    parser.add_argument("-s", "--sample", action="store_true", help="Process sample file")
    parser.add_argument("-o", "--output", default="triage_output.csv", help="Output file")
    
    args = parser.parse_args()
    
    # Determine input file
    if args.sample:
        input_file = "sample_support_tickets.csv"
    else:
        input_file = args.input_file
    
    if not Path(input_file).exists():
        print(f"❌ Error: {input_file} not found")
        sys.exit(1)
    
    print(f"🚀 Starting triage analysis on {input_file}")
    print("=" * 60)
    
    # Load and process tickets
    df = pd.read_csv(input_file)
    agent = TriageAgent()
    results = []
    
    for idx, row in df.iterrows():
        print(f"Processing ticket {idx + 1}/{len(df)}: {row.get('subject', 'No subject')[:50]}...")
        
        decision = agent.analyze_ticket(
            issue=str(row.get('issue', '')),
            subject=str(row.get('subject', '')),
            company=str(row.get('company', 'None'))
        )
        
        results.append({
            'ticket_id': idx + 1,
            'status': decision.status,
            'product_area': decision.product_area,
            'response': decision.response,
            'justification': decision.justification,
            'request_type': decision.request_type
        })
        
        print(f"  ✅ Status: {decision.status} | Type: {decision.request_type}")
        print(f"  📋 Area: {decision.product_area}")
        print()
    
    # Save results
    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output, index=False)
    
    print(f"✅ Analysis complete! Results saved to {args.output}")
    print("\n📊 Summary:")
    print(output_df['status'].value_counts())
    print(output_df['request_type'].value_counts())

if __name__ == "__main__":
    main()