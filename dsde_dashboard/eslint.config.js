{
    ('extends');
    [
        'next/core-web-vitals', // For Next.js
        'eslint:recommended',
        'plugin:@next/next/recommended',
        'plugin:react/recommended',
        'plugin:prettier/recommended',
    ],
        'plugins';
    ['prettier'], 'rules';
    {
        ('prettier/prettier');
        'error', // Marks Prettier issues as errors
            'react/react-in-jsx-scope';
        ('off'); // Next.js doesn't need React import in scope
    }
}
