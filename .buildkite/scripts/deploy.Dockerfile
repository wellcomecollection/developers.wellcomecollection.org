FROM public.ecr.aws/docker/library/node:16-alpine

RUN apk add --update git openssh
