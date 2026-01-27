"""Shared configuration for sales mix pipeline."""

# Refund notes that should be treated as valid sales.
KEEP_REFUND_PATTERNS = (
    r"panda",
    r"hungry panda",
    r"\bhp\b",
    r"\bhp\d+\b",
    r"\bhp\s*\d+\b",
)

# Item name patterns to exclude from product mix outputs.
EXCLUDE_ITEM_PATTERNS = (
    r"\btips?\b",
    r"boba tea tote bag",
    r"cupe tote bag",
    r"coaster",
    r"mosa hat",
    r"free drink",
    r"custom amount",
)

# Item name snippet for the "fun" hourly chart.
FEATURED_ITEM_QUERY = "TGY Special"
