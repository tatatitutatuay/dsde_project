'use server';

export async function extractKeywords(abstract) {
    try {
        const response = await fetch(
            'https://model-backend-f7d12ebf4930.herokuapp.com/api/extract',
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ abstract: abstract }),
                cache: 'no-store',
            }
        );

        if (!response.ok) {
            throw new Error('Failed to extract keywords');
        }

        return await response.json();
    } catch (error) {
        console.error('Keyword extraction error:', error);
        return {
            error:
                error instanceof Error
                    ? error.message
                    : 'An unknown error occurred',
        };
    }
}