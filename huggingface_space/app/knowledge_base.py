"""Tiny in-memory knowledge base for the demo RAG app: a company FAQ bot."""

DOCS = [
    {
        "id": "doc1",
        "title": "Refund Policy",
        "text": (
            "Customers can request a full refund within 30 days of purchase. "
            "After 30 days, only store credit is issued. Refunds are processed "
            "within 5-7 business days to the original payment method."
        ),
    },
    {
        "id": "doc2",
        "title": "Shipping Times",
        "text": (
            "Standard shipping takes 5-7 business days within the US. "
            "Express shipping takes 2-3 business days and costs an extra $15. "
            "International shipping takes 10-15 business days."
        ),
    },
    {
        "id": "doc3",
        "title": "Account Deletion",
        "text": (
            "Users can delete their account from Settings > Privacy > Delete Account. "
            "Account deletion is permanent after a 14-day grace period, during which "
            "the account can be restored by logging back in."
        ),
    },
    {
        "id": "doc4",
        "title": "Password Reset",
        "text": (
            "To reset a password, click 'Forgot Password' on the login page and "
            "enter your registered email. A reset link is valid for 1 hour. "
            "If the email doesn't arrive, check the spam folder before contacting support."
        ),
    },
    {
        "id": "doc5",
        "title": "Subscription Plans",
        "text": (
            "We offer three plans: Free (limited features), Pro at $12/month "
            "(full features, single user), and Team at $9/user/month (minimum 3 users, "
            "includes admin controls). All paid plans include a 14-day free trial."
        ),
    },
    {
        "id": "doc6",
        "title": "Data Export",
        "text": (
            "Users on Pro and Team plans can export their data as CSV or JSON from "
            "Settings > Data > Export. Free plan users can export CSV only, once per month."
        ),
    },
]
