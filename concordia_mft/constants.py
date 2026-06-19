"""Experiment constants: news topics, forum variants, and prompts."""

import concordia.prefabs.game_master as game_master_prefabs

MFT_KEYS = ['care', 'fairness', 'loyalty', 'authority', 'purity']

NEWS_TOPICS = [
    {
        'id': 'hospital_donation',
        'title': "Children's Hospital Receives Record Donation and Expands Treatment for Rare Diseases",
        'body': 'A nationwide campaign raised millions for a children\'s hospital, allowing the opening of new wings and free treatment for hundreds of vulnerable children.',
        'topic_mft': {'care': 5.0, 'fairness': 4.0, 'loyalty': 3.0, 'authority': 2.0, 'purity': 2.0},
    },
    {
        'id': 'congress_corruption',
        'title': 'Congressman Uses Public Funds for Personal Trip and Faces Investigation',
        'body': 'Documents indicate that resources allocated to the congressional office were used for personal expenses abroad.',
        'topic_mft': {'care': 2.0, 'fairness': 5.0, 'loyalty': 2.0, 'authority': 4.0, 'purity': 3.0},
    },
    {
        'id': 'school_anthem',
        'title': 'School Reintroduces Daily National Anthem and Stricter Discipline Rules',
        'body': 'The administration states that the measure aims to reinforce respect, order, and civic values among students.',
        'topic_mft': {'care': 1.0, 'fairness': 2.0, 'loyalty': 5.0, 'authority': 5.0, 'purity': 3.0},
    },
    {
        'id': 'religious_marriage',
        'title': 'Religious Leader Criticizes School Campaign Supporting Same-Sex Marriage',
        'body': 'During a public event, a religious leader stated that schools should prioritize academic content and preserve traditional family values.',
        'topic_mft': {'care': 2.0, 'fairness': 3.0, 'loyalty': 4.0, 'authority': 5.0, 'purity': 5.0},
    },
    {
        'id': 'drought_climate',
        'title': 'Historic Drought Forces City to Ration Water and Suspend Classes',
        'body': 'Reservoirs reached critical levels after months without rain. Experts linked the extreme conditions to the advance of climate change.',
        'topic_mft': {'care': 5.0, 'fairness': 4.0, 'loyalty': 2.0, 'authority': 2.0, 'purity': 3.0},
    },
    {
        'id': 'gun_law',
        'title': 'Congress Approves Bill Tightening Rules for Gun Purchase and Ownership',
        'body': 'Lawmakers approved new requirements for firearms acquisition, including training, psychological evaluations, and tracking measures. Supporters argue the policy could reduce deaths and accidents.',
        'topic_mft': {'care': 4.0, 'fairness': 4.0, 'loyalty': 3.0, 'authority': 4.0, 'purity': 2.0},
    },
]

FORUM_VARIANTS = [
    {'name': 'Rural News', 'forum_mft': {'care': 3.0, 'fairness': 3.0, 'loyalty': 3.0, 'authority': 3.0, 'purity': 3.0}},
    {'name': 'Voz Conservadora', 'forum_mft': {'care': 3.0, 'fairness': 3.0, 'loyalty': 4.5, 'authority': 4.5, 'purity': 4.5}},
    {'name': 'Voz Progressista', 'forum_mft': {'care': 4.5, 'fairness': 4.5, 'loyalty': 2.0, 'authority': 2.0, 'purity': 2.0}},
]

FORUM_CALL_TO_ACTION = (
    game_master_prefabs.async_social_media.DEFAULT_CALL_TO_ACTION
    + '\n\nStrict engagement policy (mandatory):\n'
    '- share: ONLY if you STRONGLY endorse the post and actively want it spread; '
    'never for politeness, visibility, mild agreement, neutrality, or disagreement.\n'
    '- upvote: ONLY for clear agreement with that exact item; never when you '
    'disagree or oppose it.\n'
    '- downvote: when the item conflicts with your core beliefs or offends a '
    'foundation that matters to you (weight higher foundations more).\n'
    '- reply: when you STRONGLY disagree and want to contest in public; use this '
    'instead of silent agreement signals.\n'
    '- If you oppose the content: NEVER share or upvote; use downvote and/or reply, '
    'or ignore.\n'
)
