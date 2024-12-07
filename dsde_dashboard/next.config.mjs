import webpack from 'webpack';

const nextConfig = {
    webpack: (config) => {
        config.plugins.push(
            new webpack.DefinePlugin({
                self: 'typeof window !== "undefined" ? window : {}', // Define `self` conditionally
            })
        );
        return config;
    },
    eslint: {
        ignoreDuringBuilds: true, // Skip ESLint checks during builds
    },
};

export default nextConfig;
