ARG TARGET_ENV=staging
FROM ghcr.io/salp-bv/salp-image-processor:${TARGET_ENV}

ENV PORT 8000
EXPOSE 8000

