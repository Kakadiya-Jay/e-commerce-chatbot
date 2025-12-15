from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.index import LocalIndex

encoder = HuggingFaceEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")

faq = Route(
    name="FAQ Route",
    score_threshold=0.3,
    utterances=[
        "What is the return policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment methods are accepted?",
        "How long does it take to process a refund?",
    ],
)

sql = Route(
    name="SQL Route",
    score_threshold=0.3,
    utterances=[
        "I want to buy nike shoes that have 50% discount.",
        "Are there any shoes under Rs. 3000?",
        "Do you have formal shoes in size 9?",
        "Are there any Puma shoes on sale?",
        "What is the price of puma running shoes?",
    ],
)

index = LocalIndex()

router = SemanticRouter(
    encoder=encoder,
    index=index,
)
router.add(routes=[faq, sql])

if __name__ == "__main__":
    user_query = "Can I return a product if I'm not satisfied?"
    selected_route = router(user_query)
    print(f"Selected Route: {selected_route.name}")
    # Optional: See exactly how confident the router was
    if hasattr(selected_route, "similarity_score"):
        print(f"Confidence Score: {selected_route.similarity_score}")
