import webpack from 'webpack';

const nextConfig = {
    webpack: (config, { isServer }) => {
        if (!isServer) {
            config.cache = false;
        }
        return config;
    },
    eslint: {
        ignoreDuringBuilds: true, // Skip ESLint checks during builds
    },
};

export default nextConfig;
