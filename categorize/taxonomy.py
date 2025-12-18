"""Category taxonomy and keyword rules for MCP Tool Classification."""

from typing import Dict, List, Tuple, Optional

# 14 Major Categories with Subcategories
CATEGORY_TAXONOMY: Dict[str, List[str]] = {
    "Communication": ["Email", "Messenger", "Social_Media", "Video_Conference"],
    "Development": ["Version_Control", "IDE_Editor", "Testing_Debug", "Build_Deploy", "Code_Analysis"],
    "Database": ["SQL", "NoSQL", "Vector_DB", "Graph_DB"],
    "Cloud_Infrastructure": ["AWS", "GCP", "Azure", "Other_Cloud"],
    "Productivity": ["Project_Management", "Note_Taking", "Calendar", "Office_Suite"],
    "Finance": ["Cryptocurrency", "Stock_Trading", "Payment", "Banking"],
    "AI_ML": ["LLM_Provider", "Image_Generation", "Memory_RAG", "Reasoning", "Embedding"],
    "Media_Content": ["Video", "Image", "Audio_Music", "2D_Design"],
    "Search_Data": ["Web_Search", "Scraping", "News_Feed", "Academic"],
    "System_Utility": ["Filesystem", "Shell_Terminal", "Time_Date", "Network", "Process"],
    "Business_Enterprise": ["CRM", "Marketing", "HR", "E_commerce"],
    "Security_Auth": ["OAuth", "2FA_TOTP", "Encryption", "PenTest", "Vulnerability"],
    "Design_Engineering": ["3D_Modeling", "CAD", "3D_Printing", "Geometry", "Manufacturing"],
    "Specialized": ["Healthcare", "Legal", "Education", "Travel", "Gaming", "IoT", "Home_Automation", "Translation", "Regional"],
}

# Keyword rules for classification
# Format: (major_category, minor_category): {exact_match: [...], contains: [...], description_keywords: [...]}
KEYWORD_RULES: Dict[Tuple[str, str], Dict[str, List[str]]] = {
    # ==================== Communication ====================
    ("Communication", "Email"): {
        "exact_match": ["gmail", "outlook", "fastmail", "protonmail", "mailchimp", "sendgrid", "mailgun", "postmark"],
        "contains": ["mail", "email", "imap", "smtp"],
        "description_keywords": ["send email", "inbox", "compose mail", "email message"],
    },
    ("Communication", "Messenger"): {
        "exact_match": ["slack", "discord", "telegram", "whatsapp", "teams", "wechat", "mattermost", "rocket.chat", "zulip"],
        "contains": ["chat", "messenger", "messaging"],
        "description_keywords": ["send message", "chat", "direct message", "channel"],
    },
    ("Communication", "Social_Media"): {
        "exact_match": ["twitter", "linkedin", "facebook", "instagram", "reddit", "bluesky", "tiktok", "mastodon", "threads", "x"],
        "contains": ["social", "tweet", "post"],
        "description_keywords": ["social media", "tweet", "post", "follow", "timeline"],
    },
    ("Communication", "Video_Conference"): {
        "exact_match": ["zoom", "meet", "webex", "jitsi", "whereby", "loom"],
        "contains": ["video call", "conference", "meeting"],
        "description_keywords": ["video call", "meeting", "conference", "screen share"],
    },

    # ==================== Development ====================
    ("Development", "Version_Control"): {
        "exact_match": ["github", "gitlab", "bitbucket", "gitea", "gogs"],
        "contains": ["git"],
        "exclude": ["digital", "legit", "digit"],  # false positive prevention
        "description_keywords": ["repository", "commit", "pull request", "branch", "merge", "clone"],
    },
    ("Development", "IDE_Editor"): {
        "exact_match": ["vscode", "cursor", "vim", "neovim", "xcode", "intellij", "sublime", "atom", "emacs", "jetbrains"],
        "contains": ["ide", "editor", "code editor"],
        "description_keywords": ["code editor", "syntax", "autocomplete", "lsp"],
    },
    ("Development", "Testing_Debug"): {
        "exact_match": ["playwright", "selenium", "puppeteer", "jest", "pytest", "cypress", "mocha", "vitest"],
        "contains": ["test", "debug", "e2e", "unit test"],
        "description_keywords": ["test", "debug", "assertion", "coverage", "mock"],
    },
    ("Development", "Build_Deploy"): {
        "exact_match": ["docker", "kubernetes", "jenkins", "circleci", "travis", "vercel", "netlify", "railway", "heroku", "render"],
        "contains": ["ci", "cd", "deploy", "build", "container", "k8s"],
        "description_keywords": ["deploy", "build", "container", "pipeline", "ci/cd"],
    },
    ("Development", "Code_Analysis"): {
        "exact_match": ["eslint", "prettier", "sonarqube", "codeclimate", "snyk"],
        "contains": ["lint", "format", "static analysis"],
        "description_keywords": ["lint", "format", "code quality", "static analysis"],
    },

    # ==================== Database ====================
    ("Database", "SQL"): {
        "exact_match": ["mysql", "postgres", "postgresql", "sqlite", "oracle", "mariadb", "mssql", "cockroachdb", "tidb"],
        "contains": ["sql"],
        "exclude": ["nosql"],
        "description_keywords": ["sql", "relational", "table", "query"],
    },
    ("Database", "NoSQL"): {
        "exact_match": ["mongodb", "redis", "firestore", "dynamodb", "cassandra", "couchdb", "couchbase", "rethinkdb"],
        "contains": ["nosql", "document db", "key-value"],
        "description_keywords": ["document", "nosql", "key-value", "cache"],
    },
    ("Database", "Vector_DB"): {
        "exact_match": ["pinecone", "chroma", "qdrant", "milvus", "weaviate", "faiss", "pgvector", "lancedb"],
        "contains": ["vector", "embedding db"],
        "description_keywords": ["vector", "embedding", "similarity search", "semantic"],
    },
    ("Database", "Graph_DB"): {
        "exact_match": ["neo4j", "arangodb", "dgraph", "tigergraph", "neptune"],
        "contains": ["graph db", "knowledge graph"],
        "description_keywords": ["graph", "node", "edge", "relationship", "cypher"],
    },

    # ==================== Cloud Infrastructure ====================
    ("Cloud_Infrastructure", "AWS"): {
        "exact_match": ["aws", "amazon", "s3", "ec2", "lambda", "bedrock", "sagemaker", "cloudwatch"],
        "contains": ["aws", "amazon web"],
        "description_keywords": ["aws", "amazon", "s3", "ec2", "lambda"],
    },
    ("Cloud_Infrastructure", "GCP"): {
        "exact_match": ["gcp", "bigquery", "vertex", "cloud run", "firebase"],
        "contains": ["google cloud", "gcp"],
        "description_keywords": ["google cloud", "bigquery", "vertex", "gcp"],
    },
    ("Cloud_Infrastructure", "Azure"): {
        "exact_match": ["azure", "cosmos", "blob storage"],
        "contains": ["azure", "microsoft cloud"],
        "description_keywords": ["azure", "microsoft cloud"],
    },
    ("Cloud_Infrastructure", "Other_Cloud"): {
        "exact_match": ["cloudflare", "digitalocean", "linode", "vultr", "hetzner", "supabase", "neon", "planetscale"],
        "contains": ["cloud", "hosting", "serverless"],
        "description_keywords": ["cloud", "hosting", "serverless", "edge"],
    },

    # ==================== Productivity ====================
    ("Productivity", "Project_Management"): {
        "exact_match": ["jira", "asana", "trello", "linear", "clickup", "monday", "basecamp", "notion", "height"],
        "contains": ["project", "task", "kanban", "sprint"],
        "description_keywords": ["project", "task", "sprint", "kanban", "board"],
    },
    ("Productivity", "Note_Taking"): {
        "exact_match": ["obsidian", "evernote", "roam", "logseq", "bear", "apple notes", "notion"],
        "contains": ["note", "wiki", "knowledge base"],
        "description_keywords": ["note", "wiki", "markdown", "knowledge"],
    },
    ("Productivity", "Calendar"): {
        "exact_match": ["google calendar", "calendly", "cal.com", "cron", "fantastical"],
        "contains": ["calendar", "schedule", "appointment"],
        "description_keywords": ["calendar", "schedule", "event", "appointment", "booking"],
    },
    ("Productivity", "Office_Suite"): {
        "exact_match": ["excel", "word", "powerpoint", "google sheets", "google docs", "airtable", "coda"],
        "contains": ["spreadsheet", "document", "presentation", "sheets"],
        "description_keywords": ["spreadsheet", "document", "presentation", "table"],
    },

    # ==================== Finance ====================
    ("Finance", "Cryptocurrency"): {
        "exact_match": ["bitcoin", "ethereum", "solana", "polygon", "binance", "coinbase", "uniswap", "opensea"],
        "contains": ["crypto", "blockchain", "defi", "nft", "web3", "wallet", "token"],
        "description_keywords": ["crypto", "blockchain", "token", "smart contract", "wallet", "defi"],
    },
    ("Finance", "Stock_Trading"): {
        "exact_match": ["yahoo finance", "bloomberg", "tradingview", "alpaca", "robinhood", "fidelity"],
        "contains": ["stock", "trading", "invest", "market", "forex"],
        "description_keywords": ["stock", "trading", "market", "price", "portfolio", "ticker"],
    },
    ("Finance", "Payment"): {
        "exact_match": ["stripe", "paypal", "square", "braintree", "adyen", "razorpay"],
        "contains": ["payment", "pay", "invoice", "billing", "checkout"],
        "description_keywords": ["payment", "charge", "invoice", "subscription", "checkout"],
    },
    ("Finance", "Banking"): {
        "exact_match": ["plaid", "yodlee", "mx"],
        "contains": ["bank", "financial", "account"],
        "description_keywords": ["bank", "account", "transaction", "balance"],
    },

    # ==================== AI/ML ====================
    ("AI_ML", "LLM_Provider"): {
        "exact_match": ["openai", "anthropic", "claude", "gemini", "ollama", "llama", "mistral", "cohere", "groq", "together"],
        "contains": ["gpt", "llm", "language model", "ai model"],
        "description_keywords": ["llm", "language model", "completion", "chat", "prompt"],
    },
    ("AI_ML", "Image_Generation"): {
        "exact_match": ["dall-e", "midjourney", "stable diffusion", "imagen", "leonardo", "flux", "replicate"],
        "contains": ["image gen", "text to image", "diffusion"],
        "description_keywords": ["image generation", "text to image", "diffusion", "generate image"],
    },
    ("AI_ML", "Memory_RAG"): {
        "exact_match": ["langchain", "llamaindex", "mem0", "zep"],
        "contains": ["memory", "rag", "retrieval", "context"],
        "description_keywords": ["memory", "rag", "retrieval", "context", "embedding"],
    },
    ("AI_ML", "Reasoning"): {
        "exact_match": ["sequential thinking", "chain of thought"],
        "contains": ["think", "thought", "reasoning", "logic"],
        "description_keywords": ["reasoning", "step by step", "logical", "problem solving", "thinking"],
    },
    ("AI_ML", "Embedding"): {
        "exact_match": ["voyage", "cohere embed", "jina"],
        "contains": ["embed", "vector"],
        "description_keywords": ["embedding", "vector", "semantic", "similarity"],
    },

    # ==================== Media Content ====================
    ("Media_Content", "Video"): {
        "exact_match": ["youtube", "vimeo", "twitch", "tiktok", "loom", "mux"],
        "contains": ["video", "stream", "media"],
        "description_keywords": ["video", "stream", "upload", "playback", "media"],
    },
    ("Media_Content", "Image"): {
        "exact_match": ["cloudinary", "imgix", "imagekit", "unsplash", "pexels"],
        "contains": ["image", "photo", "picture"],
        "exclude": ["image generation", "generate image"],
        "description_keywords": ["image", "photo", "resize", "compress", "thumbnail"],
    },
    ("Media_Content", "Audio_Music"): {
        "exact_match": ["spotify", "soundcloud", "podcast", "anchor", "descript", "elevenlabs"],
        "contains": ["audio", "music", "sound", "podcast", "voice"],
        "description_keywords": ["audio", "music", "podcast", "sound", "voice", "speech"],
    },
    ("Media_Content", "2D_Design"): {
        "exact_match": ["figma", "canva", "sketch", "adobe", "photoshop", "illustrator"],
        "contains": ["design", "graphic", "ui/ux"],
        "description_keywords": ["design", "graphic", "ui", "ux", "layout"],
    },

    # ==================== Search Data ====================
    ("Search_Data", "Web_Search"): {
        "exact_match": ["google search", "bing", "duckduckgo", "brave search", "perplexity", "exa", "tavily", "serp", "serpapi"],
        "contains": ["search", "web search"],
        "description_keywords": ["search", "query", "results", "web search"],
    },
    ("Search_Data", "Scraping"): {
        "exact_match": ["firecrawl", "browserless", "apify", "scrapy", "cheerio", "jina reader"],
        "contains": ["scrape", "crawl", "fetch", "extract"],
        "description_keywords": ["scrape", "crawl", "extract", "parse", "html"],
    },
    ("Search_Data", "News_Feed"): {
        "exact_match": ["hackernews", "rss", "feedly", "newsapi"],
        "contains": ["news", "rss", "feed"],
        "description_keywords": ["news", "feed", "article", "headline"],
    },
    ("Search_Data", "Academic"): {
        "exact_match": ["arxiv", "scholar", "pubmed", "semantic scholar", "crossref", "unpaywall"],
        "contains": ["paper", "research", "academic", "journal"],
        "description_keywords": ["paper", "research", "citation", "journal", "academic"],
    },

    # ==================== System Utility ====================
    ("System_Utility", "Filesystem"): {
        "exact_match": ["filesystem", "fs"],
        "contains": ["file", "directory", "folder", "storage", "disk"],
        "description_keywords": ["file", "directory", "read", "write", "path"],
    },
    ("System_Utility", "Shell_Terminal"): {
        "exact_match": ["terminal", "bash", "powershell", "zsh", "shell"],
        "contains": ["shell", "terminal", "cli", "command"],
        "description_keywords": ["shell", "terminal", "command", "execute", "bash"],
    },
    ("System_Utility", "Time_Date"): {
        "exact_match": ["time", "datetime", "clock", "timezone"],
        "contains": ["time", "date", "clock", "timezone"],
        "description_keywords": ["time", "date", "timezone", "timestamp", "clock"],
    },
    ("System_Utility", "Network"): {
        "exact_match": ["dns", "http", "network", "proxy", "vpn", "curl", "fetch"],
        "contains": ["network", "http", "dns", "ip", "proxy"],
        "description_keywords": ["network", "http", "request", "dns", "ip"],
    },
    ("System_Utility", "Process"): {
        "exact_match": ["process", "pm2", "supervisor"],
        "contains": ["process", "daemon", "service"],
        "description_keywords": ["process", "run", "execute", "spawn"],
    },

    # ==================== Business Enterprise ====================
    ("Business_Enterprise", "CRM"): {
        "exact_match": ["hubspot", "salesforce", "pipedrive", "attio", "close", "freshsales", "zoho crm"],
        "contains": ["crm", "customer"],
        "description_keywords": ["crm", "customer", "lead", "contact", "deal"],
    },
    ("Business_Enterprise", "Marketing"): {
        "exact_match": ["google analytics", "mixpanel", "amplitude", "segment", "mailerlite", "convertkit"],
        "contains": ["marketing", "analytics", "seo", "campaign"],
        "description_keywords": ["marketing", "campaign", "analytics", "seo", "conversion"],
    },
    ("Business_Enterprise", "HR"): {
        "exact_match": ["workday", "bamboohr", "gusto", "rippling", "lever", "greenhouse"],
        "contains": ["hr", "recruit", "hiring", "employee"],
        "description_keywords": ["hr", "employee", "hiring", "recruit", "payroll"],
    },
    ("Business_Enterprise", "E_commerce"): {
        "exact_match": ["shopify", "woocommerce", "magento", "bigcommerce", "stripe"],
        "contains": ["ecommerce", "shop", "store", "product", "inventory"],
        "description_keywords": ["shop", "product", "order", "inventory", "cart"],
    },

    # ==================== Security Auth ====================
    ("Security_Auth", "OAuth"): {
        "exact_match": ["auth0", "okta", "clerk", "firebase auth", "supabase auth", "better auth"],
        "contains": ["oauth", "auth", "sso", "identity"],
        "description_keywords": ["oauth", "authentication", "sso", "identity", "login"],
    },
    ("Security_Auth", "2FA_TOTP"): {
        "exact_match": ["totp", "2fa", "authy", "duo"],
        "contains": ["2fa", "totp", "otp", "mfa"],
        "description_keywords": ["2fa", "totp", "otp", "two-factor", "mfa"],
    },
    ("Security_Auth", "Encryption"): {
        "exact_match": ["vault", "1password", "bitwarden", "lastpass"],
        "contains": ["encrypt", "decrypt", "secret", "vault", "password"],
        "description_keywords": ["encrypt", "decrypt", "secret", "key", "vault"],
    },
    ("Security_Auth", "PenTest"): {
        "exact_match": ["kali", "metasploit", "burp", "nmap", "ctf"],
        "contains": ["pentest", "security test", "vulnerability scan", "ctf"],
        "description_keywords": ["pentest", "vulnerability", "exploit", "security test", "ctf"],
    },
    ("Security_Auth", "Vulnerability"): {
        "exact_match": ["snyk", "dependabot", "trivy", "cve"],
        "contains": ["vulnerability", "cve", "security scan"],
        "description_keywords": ["vulnerability", "cve", "security", "scan"],
    },

    # ==================== Design Engineering ====================
    ("Design_Engineering", "3D_Modeling"): {
        "exact_match": ["blender", "tripo", "spline", "sketchfab"],
        "contains": ["3d model", "mesh", "render"],
        "description_keywords": ["3d", "model", "mesh", "render", "scene"],
    },
    ("Design_Engineering", "CAD"): {
        "exact_match": ["freecad", "autocad", "solidworks", "fusion 360", "onshape"],
        "contains": ["cad", "engineering design"],
        "description_keywords": ["cad", "engineering", "design", "drawing"],
    },
    ("Design_Engineering", "3D_Printing"): {
        "exact_match": ["3d printer", "octoprint", "cura", "prusa"],
        "contains": ["3d print", "slicer", "gcode"],
        "description_keywords": ["3d print", "slicer", "gcode", "filament"],
    },
    ("Design_Engineering", "Geometry"): {
        "exact_match": ["asymptote", "geogebra"],
        "contains": ["geometry", "math visual"],
        "description_keywords": ["geometry", "shape", "coordinate", "vector"],
    },
    ("Design_Engineering", "Manufacturing"): {
        "exact_match": ["cnc", "laser"],
        "contains": ["manufacturing", "cnc", "fabrication"],
        "description_keywords": ["manufacturing", "cnc", "machine", "fabrication"],
    },

    # ==================== Specialized ====================
    ("Specialized", "Healthcare"): {
        "exact_match": ["health", "medical", "fhir", "epic"],
        "contains": ["health", "medical", "clinical", "patient"],
        "description_keywords": ["health", "medical", "patient", "clinical", "diagnosis"],
    },
    ("Specialized", "Legal"): {
        "exact_match": ["clio", "legalzoom"],
        "contains": ["legal", "law", "court", "contract"],
        "description_keywords": ["legal", "law", "contract", "compliance"],
    },
    ("Specialized", "Education"): {
        "exact_match": ["canvas", "moodle", "coursera", "udemy"],
        "contains": ["education", "learn", "course", "student", "teacher"],
        "description_keywords": ["education", "course", "learn", "student", "quiz"],
    },
    ("Specialized", "Travel"): {
        "exact_match": ["airbnb", "booking", "expedia", "tripadvisor", "skyscanner"],
        "contains": ["travel", "flight", "hotel", "booking"],
        "description_keywords": ["travel", "flight", "hotel", "booking", "trip"],
    },
    ("Specialized", "Gaming"): {
        "exact_match": ["steam", "xbox", "playstation", "nintendo", "epic games", "twitch"],
        "contains": ["game", "gaming", "esport"],
        "description_keywords": ["game", "gaming", "player", "score", "achievement"],
    },
    ("Specialized", "IoT"): {
        "exact_match": ["arduino", "raspberry", "esp32", "zigbee", "mqtt"],
        "contains": ["iot", "sensor", "device", "smart"],
        "description_keywords": ["iot", "sensor", "device", "smart home", "automation"],
    },
    ("Specialized", "Home_Automation"): {
        "exact_match": ["home assistant", "homekit", "smartthings", "alexa", "google home"],
        "contains": ["home automation", "smart home"],
        "description_keywords": ["home", "automation", "smart", "control", "light"],
    },
    ("Specialized", "Translation"): {
        "exact_match": ["deepl", "google translate"],
        "contains": ["translate", "translation", "localization"],
        "description_keywords": ["translate", "translation", "language", "localize"],
    },
    ("Specialized", "Regional"): {
        "exact_match": [],
        "contains": ["korean", "chinese", "japanese", "german", "french", "spanish"],
        "description_keywords": ["regional", "local", "country specific"],
    },
}


def get_all_major_categories() -> List[str]:
    """Get list of all major categories."""
    return list(CATEGORY_TAXONOMY.keys())


def get_subcategories(major_category: str) -> List[str]:
    """Get subcategories for a major category."""
    return CATEGORY_TAXONOMY.get(major_category, [])


def get_all_subcategories() -> List[Tuple[str, str]]:
    """Get list of all (major, minor) category tuples."""
    result = []
    for major, minors in CATEGORY_TAXONOMY.items():
        for minor in minors:
            result.append((major, minor))
    return result


def get_taxonomy_description() -> str:
    """Get a formatted description of the taxonomy for LLM prompts."""
    lines = ["Category Taxonomy (14 Major Categories):"]
    for major, minors in CATEGORY_TAXONOMY.items():
        lines.append(f"  {major}: {', '.join(minors)}")
    return "\n".join(lines)
