import { defineConfig } from 'eslint-define-config';

export default defineConfig([
    {
        files: ['**/*.js', '**/*.jsx', '**/*.ts', '**/*.tsx'], // Specify the files to lint
        languageOptions: {
            parserOptions: {
                ecmaVersion: 2023,
                sourceType: 'module',
                ecmaFeatures: {
                    jsx: true,
                },
            },
        },
        plugins: {
            prettier: require('eslint-plugin-prettier'),
            react: require('eslint-plugin-react'),
            '@next/next': require('@next/eslint-plugin-next'),
        },
        rules: {
            'prettier/prettier': 'error',
            'react/react-in-jsx-scope': 'off',
        },
    },
]);
