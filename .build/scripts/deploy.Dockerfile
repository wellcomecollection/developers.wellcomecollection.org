FROM public.ecr.aws/docker/library/node:18-alpine

RUN apk add --update git openssh
