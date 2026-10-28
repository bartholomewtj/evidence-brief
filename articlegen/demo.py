"""A built-in sample so `articlegen demo` can render the design without any API calls.

The prose here is illustrative and the citations map to the sample sources below;
it exists to exercise the renderer, not as a real piece of journalism.
`SAMPLE_BRIEFING` is the default artefact; `SAMPLE_ARTICLE` is the `--long` Review.
"""

from .sources import Paper

SAMPLE_PAPERS = [
    Paper(
        title="The memory function of sleep",
        abstract="Sleep consolidates newly encoded memories via hippocampal replay.",
        year=2010,
        authors=["Susanne Diekelmann", "Jan Born"],
        venue="Nature Reviews Neuroscience",
        citation_count=3200,
        doi="10.1038/nrn2762",
        source="demo",
    ),
    Paper(
        title="Sleep-dependent memory consolidation",
        abstract="A review of behavioral and physiological evidence linking sleep to memory.",
        year=2005,
        authors=["Robert Stickgold"],
        venue="Nature",
        citation_count=2100,
        doi="10.1038/nature04286",
        source="demo",
    ),
    Paper(
        title="Sleep drives metabolite clearance from the adult brain",
        abstract="During sleep the glymphatic system clears metabolic waste including amyloid-beta.",
        year=2013,
        authors=["Lulu Xie", "Hongyi Kang", "Qiwu Xu", "et al."],
        venue="Science",
        citation_count=2800,
        doi="10.1126/science.1241224",
        source="demo",
    ),
    Paper(
        title="Interindividual differences in the effects of sleep deprivation",
        abstract="People differ reliably and heritably in vulnerability to sleep loss.",
        year=2004,
        authors=["Hans Van Dongen", "David Dinges"],
        venue="Sleep",
        citation_count=900,
        doi="10.1093/sleep/27.3.423",
        source="demo",
    ),
    Paper(
        title="Targeted memory reactivation during sleep",
        abstract="Cueing learned material during slow-wave sleep improves later recall.",
        year=2013,
        authors=["Delphine Oudiette", "Ken Paller"],
        venue="Trends in Cognitive Sciences",
        citation_count=640,
        doi="10.1016/j.tics.2013.01.006",
        source="demo",
    ),
    Paper(
        title="The sleep-deprived human brain",
        abstract="Sleep loss degrades attention, emotional regulation and memory encoding.",
        year=2017,
        authors=["Adam Krause", "Eti Simon", "Bryce Mander", "et al."],
        venue="Nature Reviews Neuroscience",
        citation_count=1100,
        doi="10.1038/nrn.2017.55",
        source="demo",
    ),
    Paper(
        title="Sleep and the price of plasticity",
        abstract="A synaptic homeostasis account of why sleep is required for learning.",
        year=2014,
        authors=["Giulio Tononi", "Chiara Cirelli"],
        venue="Neuron",
        citation_count=2400,
        doi="10.1016/j.neuron.2013.12.025",
        source="demo",
    ),
    Paper(
        title="Circadian regulation of glymphatic function",
        abstract="Clearance in rodents varies with circadian phase independently of sleep state.",
        year=2020,
        authors=["Lauren Hablitz", "Virginia Pla", "Michael Giannetto", "et al."],
        venue="Nature Communications",
        citation_count=430,
        doi="10.1038/s41467-020-18115-2",
        source="demo",
    ),
]

# What `curate_sources` would return for these papers — drives Table 1 and Fig. 1.
SAMPLE_CURATION = {
    "relevance": {
        1: "direct", 2: "direct", 3: "direct", 4: "related",
        5: "related", 6: "related", 7: "tangential", 8: "tangential",
    },
    "most_relevant_index": 3,
    "counts": {"direct": 3, "related": 3, "tangential": 2},
}

SAMPLE_PROVENANCE = {
    "queries": ["sleep memory consolidation", "glymphatic clearance sleep", "sleep deprivation individual differences"],
    "core_entity": "sleep and the brain",
    # Named so the demo shows the Methods section a real run produces. Methods
    # names only the databases that answered and infers nothing, so leaving this
    # out would make the sample demonstrate the unrecorded-search wording.
    "databases": ["Semantic Scholar Graph API", "OpenAlex", "Europe PMC"],
    "model": "demo build — no model call",
}

SAMPLE_BRIEFING = {
    "form": "briefing",
    "title": "Sleep, memory consolidation and metabolite clearance in the brain",
    "question": (
        "What does published evidence indicate the sleeping brain does for newly "
        "encoded memory and for clearance of interstitial metabolites, and how far "
        "do the human data support the rodent findings?"
    ),
    "answer": (
        "Sleep supports the consolidation of new memories in people, but its role "
        "in clearing metabolites such as amyloid-beta from the brain rests mainly "
        "on rodent imaging. The memory evidence is the stronger of the two: "
        "behavioural studies in people are consistent with a consolidation role, "
        "and targeted reactivation during slow-wave sleep can bias later recall. "
        "Circadian phase may drive clearance independently of sleep state, which "
        "complicates attributing that effect to sleep as such. Vulnerability to "
        "sleep loss varies substantially between individuals, so population "
        "averages understate the range of outcomes."
    ),
    "certainty": {
        "level": "moderate",
        "reason": "Consistent human behavioural studies for memory; clearance "
        "rests on rodent imaging alone.",
    },
    "keywords": [
        "sleep", "memory consolidation", "glymphatic system", "amyloid-beta",
        "slow-wave sleep", "sleep deprivation",
    ],
    "findings": [
        {
            "claim": "Sleep after learning improves retention, and replay during "
            "slow-wave sleep is the leading mechanism.",
            "evidence": "A review of hippocampal replay reported that during "
            "slow-wave sleep the hippocampus replays patterns recorded during "
            "preceding wakefulness, coordinated with cortical slow oscillations "
            "[1]. Participants who slept after learning retained more than those "
            "kept awake for an equivalent interval, which is consistent with a "
            "causal role rather than a mere correlation with time [2].",
            "certainty": "moderate",
        },
        {
            "claim": "Consolidation can be steered experimentally during "
            "slow-wave sleep.",
            "evidence": "Cueing specific learned material during slow-wave sleep "
            "with an odour or sound paired at encoding selectively improved later "
            "recall of the cued material [5].",
            "certainty": "moderate",
        },
        {
            "claim": "Sleep-linked clearance of amyloid-beta has been shown in "
            "mice, not in naturally sleeping people.",
            "evidence": "Two-photon imaging in mice showed that the interstitial "
            "space expands during sleep and that clearance of amyloid-beta "
            "increases accordingly [3]. Later rodent work found that glymphatic "
            "function varies with circadian phase independently of sleep state, "
            "which weakens the attribution of clearance to sleep alone [8].",
            "certainty": "low",
        },
        {
            "claim": "Vulnerability to sleep loss differs widely and stably "
            "between people.",
            "evidence": "Differences between individuals in vulnerability to "
            "sleep loss were large, stable within a person, and partly heritable, "
            "so group means may obscure a substantial minority who are severely "
            "affected [4].",
            "certainty": "moderate",
        },
    ],
    "unknowns": [
        "A human measurement of metabolite clearance that separates sleep state "
        "from circadian phase has not been identified in this set.",
        "Whether synaptic homeostasis and replay-based consolidation are "
        "complementary or competing accounts remains unresolved [7].",
        "How far rodent clearance findings carry across to people is still a "
        "hypothesis rather than an established finding [3, 8].",
    ],
    "open_these": [
        {"source_index": 3, "why": "The imaging study behind the clearance claim, "
         "with its methods and the limits of a mouse model."},
        {"source_index": 1, "why": "The clearest account of how replay is thought "
         "to move memories from hippocampus to cortex."},
        {"source_index": 5, "why": "Shows how targeted reactivation was done and "
         "how large the recall benefit was."},
    ],
    "glossary": [
        {
            "term": "Glymphatic system",
            "definition": "A proposed brain-wide route by which cerebrospinal fluid "
            "exchanges with interstitial fluid and carries away metabolic waste.",
        },
        {
            "term": "Slow-wave sleep",
            "definition": "The deepest stage of non-REM sleep, characterised by "
            "high-amplitude, low-frequency cortical oscillations.",
        },
        {
            "term": "Targeted memory reactivation",
            "definition": "Replaying a cue that was paired with material at learning "
            "while a participant sleeps, in order to bias which memories are consolidated.",
        },
    ],
    "references": [1, 2, 5, 3, 8, 4, 7],
}

SAMPLE_ARTICLE = {
    "title": "What the sleeping brain does: memory consolidation and metabolite clearance",
    "abstract": (
        "Sleep occupies roughly a third of human life, and its function has been "
        "debated for as long as it has been measured. Two decades of work in rodents "
        "and humans have converged on the view that sleep is not a passive interruption "
        "of waking but a state in which specific physiological processes run that cannot "
        "run efficiently during wakefulness. Here the evidence for two such processes is "
        "reviewed: the consolidation of newly encoded memories through coordinated "
        "hippocampal-cortical replay, and the clearance of interstitial metabolites, "
        "including amyloid-beta, through the glymphatic system. Behavioural and imaging "
        "studies in humans support a consolidation role, and targeted reactivation "
        "experiments indicate that the process can be biased experimentally. Clearance "
        "evidence remains dominated by rodent work, and recent findings suggest that "
        "circadian phase contributes independently of sleep state. Vulnerability to sleep "
        "loss varies substantially and heritably between individuals, which limits the "
        "generality of population-level estimates. Taken together, the two functions "
        "provide a mechanistic account of why sleep loss degrades cognition, and identify "
        "the human clearance literature as the principal gap."
    ),
    "keywords": [
        "sleep", "memory consolidation", "glymphatic system", "amyloid-beta",
        "slow-wave sleep", "sleep deprivation",
    ],
    "featured_study": {
        "source_index": 3,
        "method": "Two-photon imaging with tracer injection in sleeping versus waking mice.",
        "results": (
            "Interstitial space expanded during sleep, with a corresponding increase in "
            "the clearance rate of amyloid-beta."
        ),
        "limitations": (
            "An anaesthetised-mouse model; whether clearance scales the same way in "
            "naturally sleeping humans was not tested."
        ),
    },
    "sections": [
        {
            "heading": "Introduction",
            "paragraphs": [
                "Sleep is conserved across species and is defended homeostatically after "
                "loss, which implies a function that waking cannot supply [1, 2]. Two "
                "candidate functions have accumulated the most direct evidence: the "
                "stabilisation of newly encoded memory, and the removal of metabolic "
                "waste from the interstitial space of the brain [1, 3].",
                "This review considers the evidence for each, the strength of the human "
                "data supporting them, and the extent to which findings from rodent "
                "models can be carried across to people.",
            ],
        },
        {
            "heading": "Consolidation of newly encoded memory",
            "paragraphs": [
                "During slow-wave sleep the hippocampus replays patterns of activity "
                "recorded during preceding wakefulness, and this replay is coordinated "
                "with cortical slow oscillations [1]. Behavioural studies are consistent "
                "with a causal role: participants who sleep after learning show better "
                "retention than those kept awake for an equivalent interval [2].",
                "Targeted memory reactivation provides a more direct test. Cueing "
                "specific learned material during slow-wave sleep, with an odour or a "
                "sound paired with it at encoding, selectively improves later recall of "
                "the cued material [5]. These data suggest that consolidation is not "
                "merely correlated with sleep but can be biased experimentally within it.",
                "A complementary account holds that sleep serves synaptic homeostasis, "
                "renormalising the net potentiation accumulated during waking and thereby "
                "restoring the capacity to encode [7]. The two accounts are not exclusive, "
                "and the balance between them remains unresolved.",
            ],
        },
        {
            "heading": "Clearance of interstitial metabolites",
            "paragraphs": [
                "Work in mice showed that the interstitial space expands during sleep and "
                "that clearance of solutes, including amyloid-beta, increases accordingly "
                "[3]. This has been proposed as a mechanistic link between chronic sleep "
                "disruption and neurodegenerative risk.",
                "The strength of this inference should not be overstated. The primary "
                "evidence is from rodent models, and subsequent work indicates that "
                "glymphatic function varies with circadian phase independently of sleep "
                "state [8], which complicates the attribution of clearance to sleep as "
                "such. Human evidence remains indirect.",
            ],
        },
        {
            "heading": "Interindividual variation in vulnerability",
            "paragraphs": [
                "The cognitive cost of sleep restriction is not uniform. Differences "
                "between individuals in vulnerability to sleep loss are large, stable "
                "within a person across occasions, and partly heritable [4]. Population "
                "averages therefore understate the range of outcomes, and studies powered "
                "on group means may obscure a substantial minority who are severely "
                "affected.",
                "Sleep loss degrades attention, emotional regulation and the encoding of "
                "new material [6], and the relative sensitivity of these domains also "
                "appears to differ between individuals.",
            ],
        },
        {
            "heading": "Conclusions and outlook",
            "paragraphs": [
                "The evidence that sleep consolidates memory is direct, replicated in "
                "humans, and experimentally manipulable [1, 2, 5]. The evidence that "
                "sleep clears interstitial metabolites is mechanistically compelling but "
                "rests on animal models [3, 8], and the human case is not yet established.",
                "The clearest gap is a human measurement of clearance that separates the "
                "contribution of sleep state from that of circadian phase. Until such "
                "measurements exist, claims that sleep loss causes protein accumulation in "
                "the human brain should be treated as a hypothesis rather than a finding.",
            ],
        },
    ],
    "key_points": [
        "Sleep supports the stabilisation of newly encoded memory through coordinated "
        "hippocampal-cortical replay during slow-wave sleep [1, 2].",
        "Targeted reactivation during sleep selectively improves recall of the cued "
        "material, indicating that consolidation can be biased experimentally [5].",
        "Clearance of interstitial metabolites, including amyloid-beta, increases during "
        "sleep in rodent models [3]; the human evidence is indirect.",
        "Circadian phase contributes to clearance independently of sleep state, which "
        "complicates attributing clearance to sleep alone [8].",
        "Vulnerability to sleep loss is large, stable within individuals and partly "
        "heritable, so population averages understate the range of outcomes [4].",
    ],
    "glossary": [
        {
            "term": "Glymphatic system",
            "definition": "A proposed brain-wide route by which cerebrospinal fluid "
            "exchanges with interstitial fluid and carries away metabolic waste.",
        },
        {
            "term": "Slow-wave sleep",
            "definition": "The deepest stage of non-REM sleep, characterised by "
            "high-amplitude, low-frequency cortical oscillations.",
        },
        {
            "term": "Targeted memory reactivation",
            "definition": "Replaying a cue that was paired with material at learning "
            "while a participant sleeps, in order to bias which memories are consolidated.",
        },
    ],
    "references": [1, 2, 3, 5, 7, 8, 4, 6],
}
