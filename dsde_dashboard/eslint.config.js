module.exports = {
    extends: [
        'next/core-web-vitals', // Recommended Next.js rules
        'eslint:recommended',
        'plugin:@next/next/recommended',
        'plugin:react/recommended',
        'plugin:prettier/recommended', // Integrates Prettier with ESLint
    ],
    plugins: ['prettier'],
    rules: {
        'prettier/prettier': 'error', // Treat Prettier formatting issues as errors
        'react/react-in-jsx-scope': 'off', // Next.js doesn't require React imports in scope
    },
};
