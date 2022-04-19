"use strict";(self.webpackChunkdevelopers=self.webpackChunkdevelopers||[]).push([[924],{2427:function(e){e.exports=JSON.parse('{"type":"object","content":{"openapi":"3.1.0","info":{"title":"Catalogue","description":"Search our collections","version":"v2","contact":{}},"servers":[{"url":"https://api.wellcomecollection.org/catalogue/v2"}],"paths":{"/works":{"get":{"tags":["Works"],"summary":"/works","description":"Returns a paginated list of works","operationId":"getWorks","parameters":[{"name":"include","in":"query","description":"A comma-separated list of extra fields to include","schema":{"type":"string","enum":["identifiers","items","holdings","subjects","genres","contributors","production","languages","notes","images","succeededBy","precededBy","partOf","parts"]}},{"name":"items.locations.locationType","in":"query","description":"Filter by the LocationType of items on the retrieved works","schema":{"type":"string"}},{"name":"workType","in":"query","description":"Filter by the format of the searched works","schema":{"type":"string"}},{"name":"type","in":"query","description":"Filter by the type of the searched works","schema":{"type":"string","enum":["Collection","Series","Section"]}},{"name":"aggregations","in":"query","description":"What aggregated data in correlation to the results should we return.","schema":{"type":"string","enum":["workType","genres.label","production.dates","subjects.label","contributors.agent.label","items.locations.license","languages","availabilities"]}},{"name":"languages","in":"query","description":"Filter the work by language.","schema":{"type":"string"}},{"name":"genres.label","in":"query","description":"Filter the work by genre.","schema":{"type":"string"}},{"name":"subjects.label","in":"query","description":"Filter the work by subject.","schema":{"type":"string"}},{"name":"contributors.agent.label","in":"query","description":"Filter the work by contributor.","schema":{"type":"string"}},{"name":"identifiers","in":"query","description":"Filter the work by identifiers.","schema":{"type":"string"}},{"name":"items","in":"query","description":"Filter for works with items with a given canonical ID.","schema":{"type":"string"}},{"name":"items.identifiers","in":"query","description":"Filter for works with items with a given identifier.","schema":{"type":"string"}},{"name":"partOf","in":"query","description":"Filter the work by partOf relation.","schema":{"type":"string"}},{"name":"availabilities","in":"query","description":"Filter the work by availabilities.","schema":{"type":"string"}},{"name":"items.locations.accessConditions.status","in":"query","description":"Filter the work by access status.","schema":{"type":"string","enum":["open","open-with-advisory","restricted","closed","licensed-resources","unavailable","temporarily-unavailable","by-appointment","permission-required"]}},{"name":"items.locations.license","in":"query","description":"Filter the work by license.","schema":{"type":"string","enum":["cc-by","cc-by-nc","cc-by-nc-nd","cc-0","pdm","cc-by-nd","cc-by-sa","cc-by-nc-sa","ogl","opl","inc"]}},{"name":"sort","in":"query","description":"Which field to sort the results on","schema":{"type":"string","enum":["production.dates"]}},{"name":"sortOrder","in":"query","description":"The order that the results should be returned in.","schema":{"type":"string","enum":["asc","desc"]}},{"name":"production.dates.to","in":"query","description":"Return all works with a production date before and including this date.\\n\\nCan be used in conjunction with `production.dates.from` to create a range.","schema":{"type":"string"}},{"name":"production.dates.from","in":"query","description":"Return all works with a production date after and including this date.\\n\\nCan be used in conjunction with `production.dates.to` to create a range.","schema":{"type":"string"}},{"name":"query","in":"query","description":"Full-text search query, which will OR supplied terms by default.\\\\n\\\\nThe following special characters can be used to change the search behaviour:\\\\n\\\\n- \\\\\\" wraps a number of tokens to signify a phrase for searching\\\\n\\\\nTo search for any of these special characters, they should be escaped with \\\\.","schema":{"type":"string"}},{"name":"page","in":"query","description":"The page to return from the result list","schema":{"minimum":1,"type":"integer","format":"int64","default":1}},{"name":"pageSize","in":"query","description":"The number of works to return per page","schema":{"maximum":100,"minimum":1,"type":"integer","format":"int64","default":10}},{"name":"_queryType","in":"query","description":"Which query should we use search the works? Used predominantly for internal testing of relevancy. Considered Unstable.","schema":{"type":"string","enum":["MultiMatcher"]}}],"responses":{"200":{"description":"The works","content":{"*/*":{"schema":{"$ref":"#/components/schemas/WorkResultList"}}}},"400":{"description":"Bad Request Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}},"404":{"description":"Not Found Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}},"410":{"description":"Gone Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}},"500":{"description":"Internal Server Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}}}}},"/works/{id}":{"get":{"tags":["Works"],"summary":"/works/{id}","description":"Returns a single work","operationId":"getWork","parameters":[{"name":"id","in":"path","description":"The work to return","required":true,"schema":{"type":"string"}},{"name":"include","in":"query","description":"A comma-separated list of extra fields to include","schema":{"type":"string","enum":["identifiers","items","holdings","subjects","genres","contributors","production","languages","notes","images","succeededBy","precededBy","partOf","parts"]}}],"responses":{"200":{"description":"The work","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Work"}}}},"400":{"description":"Bad Request Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}},"404":{"description":"Not Found Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}},"410":{"description":"Gone Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}},"500":{"description":"Internal Server Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}}}}},"/images":{"get":{"tags":["Images"],"summary":"/images","description":"Returns a paginated list of images","operationId":"getImages","parameters":[{"name":"query","in":"query","description":"Full-text search query","schema":{"type":"string"}},{"name":"locations.license","in":"query","description":"Filter the images by license.","schema":{"type":"string","enum":["cc-by","cc-by-nc","cc-by-nc-nd","cc-0","pdm","cc-by-nd","cc-by-sa","cc-by-nc-sa","ogl","opl","inc"]}},{"name":"source.contributors.agent.label","in":"query","description":"Filter the images by the source works\' contributors","schema":{"type":"string"}},{"name":"source.genres.label","in":"query","description":"Filter the images by the source works\' genres","schema":{"type":"string"}},{"name":"colors","in":"query","description":"Filter the images by colors.","schema":{"type":"string"}},{"name":"include","in":"query","description":"A comma-separated list of extra fields to include","schema":{"type":"string","enum":["source.contributors","source.languages","source.genres"]}},{"name":"aggregations","in":"query","description":"What aggregated data in correlation to the results should we return.","schema":{"type":"string","enum":["locations.license","source.contributors.agent.label","source.genres.label"]}},{"name":"page","in":"query","description":"The page to return from the result list","schema":{"minimum":1,"type":"integer","format":"int64","default":1}},{"name":"pageSize","in":"query","description":"The number of images to return per page","schema":{"maximum":100,"minimum":1,"type":"integer","format":"int64","default":10}}],"responses":{"200":{"description":"The images","content":{"*/*":{"schema":{"$ref":"#/components/schemas/ImageResultList"}}}},"400":{"description":"Bad Request Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}},"404":{"description":"Not Found Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}},"410":{"description":"Gone Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}},"500":{"description":"Internal Server Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}}}}},"/images/{id}":{"get":{"tags":["Images"],"summary":"/images/{id}","description":"Returns a single image","operationId":"getImage","parameters":[{"name":"id","in":"path","description":"The image to return","required":true,"schema":{"type":"string"}},{"name":"include","in":"query","description":"A comma-separated list of extra fields to include","schema":{"type":"string","enum":["visuallySimilar","withSimilarFeatures","withSimilarColors","source.contributors","source.languages","source.genres"]}}],"responses":{"200":{"description":"The image","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Image"}}}},"400":{"description":"Bad Request Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}},"404":{"description":"Not Found Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}},"410":{"description":"Gone Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}},"500":{"description":"Internal Server Error","content":{"*/*":{"schema":{"$ref":"#/components/schemas/Error"}}}}}}}},"components":{"schemas":{"AccessCondition":{"title":"AccessCondition","type":"object","description":"Information about any access restrictions placed on the work","properties":{"method":{"$ref":"#/components/schemas/AccessMethod"},"status":{"$ref":"#/components/schemas/AccessStatus"},"terms":{"type":"string"},"note":{"type":"string"},"type":{"type":"string","default":"AccessCondition"}}},"AccessMethod":{"title":"AccessMethod","type":"object","properties":{"id":{"type":"string","enum":["online-request","manual-request","not-requestable","view-online","open-shelves"]},"label":{"type":"string","enum":["Online request","Manual request","Not requestable","View online","Open shelves"]},"type":{"type":"string","default":"AccessMethod"}},"examples":[{"id":"online-request","label":"Online request","type":"AccessMethod"}]},"AccessStatus":{"title":"AccessStatus","type":"object","properties":{"id":{"type":"string"},"label":{"type":"string"},"type":{"type":"string","default":"AccessStatus"}}},"Agent":{"title":"Agent","properties":{"id":{"type":"string"},"label":{"type":"string"},"type":{"type":"string"}},"type":"object","description":"What are the actual available types here?"},"Aggregation":{"title":"Aggregation","type":"object","description":"An aggregation over the results.","properties":{"buckets":{"type":"array","items":{"$ref":"#/components/schemas/AggregationBucket"}},"type":{"type":"string"}}},"AggregationBucket":{"title":"AggregationBucket","type":"object","description":"An individual bucket within an aggregation.","properties":{"data":{"discriminator":{"propertyName":"type"},"oneOf":[{"$ref":"#/components/schemas/Format"},{"$ref":"#/components/schemas/Period"},{"$ref":"#/components/schemas/Genre"},{"$ref":"#/components/schemas/Subject"},{"$ref":"#/components/schemas/Agent"},{"$ref":"#/components/schemas/Language"},{"$ref":"#/components/schemas/License"},{"$ref":"#/components/schemas/Availability"}]},"count":{"type":"integer","description":"The count of how often this data occurs in this set of results.","format":"int32"},"type":{"type":"string"}}},"Availability":{"title":"Availability","type":"object","properties":{"id":{"type":"string"},"label":{"type":"string"},"type":{"type":"string"}},"description":"Ways in which the work is available to access"},"Concept":{"title":"Concept","type":"object","description":"What are the actual available types here? - Concept, Period, Place","properties":{"id":{"type":"string"},"label":{"type":"string"},"identifiers":{"type":"array","items":{"$ref":"#/components/schemas/Identifier"}},"type":{"type":"string"}}},"ContributionRole":{"title":"ContributionRole","type":"object","properties":{"label":{"type":"string"},"type":{"type":"string"}},"description":"A contribution role"},"Contributor":{"title":"Contributor","type":"object","description":"A contributor","properties":{"agent":{"$ref":"#/components/schemas/Agent"},"roles":{"type":"array","items":{"$ref":"#/components/schemas/ContributionRole"}},"type":{"type":"string"}}},"DigitalLocation":{"title":"DigitalLocation","type":"object","description":"A digital location that provides access to an item","properties":{"locationType":{"$ref":"#/components/schemas/LocationType"},"url":{"type":"string","description":"The URL of the digital asset."},"credit":{"type":"string","description":"Who to credit the image to"},"linkText":{"type":"string","description":"Text that can be used when linking to the item - for example, \'View this journal\' rather than the raw URL"},"license":{"$ref":"#/components/schemas/License"},"accessConditions":{"type":"array","items":{"$ref":"#/components/schemas/AccessCondition"}},"type":{"type":"string"}}},"Error":{"title":"Error","type":"object","properties":{"errorType":{"type":"string","description":"The type of error","enum":["http"]},"httpStatus":{"type":"integer","description":"The HTTP response status code","format":"int32"},"label":{"type":"string","description":"The title or other short name of the error"},"description":{"type":"string","description":"The specific error"},"type":{"type":"string"}}},"Format":{"title":"Format","type":"object","properties":{"id":{"type":"string"},"label":{"type":"string"},"type":{"type":"string"}},"description":"A broad, top-level description of the form of a work: namely, whether it is a printed book, archive, painting, photograph, moving image, etc."},"Genre":{"title":"Genre","type":"object","description":"A genre","properties":{"label":{"type":"string","description":"A label given to a thing."},"concepts":{"type":"array","items":{"$ref":"#/components/schemas/Concept"}},"type":{"type":"string"}}},"Holdings":{"title":"Holdings","type":"object","description":"A collection of materials owned by the library.","properties":{"note":{"type":"string","description":"Additional information about the holdings."},"enumeration":{"type":"array","items":{"type":"string","description":"A list of individual issues or parts that make up the holdings."}},"location":{"description":"The location of the holdings","discriminator":{"propertyName":"type"},"oneOf":[{"$ref":"#/components/schemas/DigitalLocation"},{"$ref":"#/components/schemas/PhysicalLocation"}]},"type":{"type":"string"}}},"Identifier":{"title":"Identifier","type":"object","description":"A unique system-generated identifier that governs interaction between systems and is regarded as canonical within the Wellcome data ecosystem.","properties":{"identifierType":{"$ref":"#/components/schemas/IdentifierType"},"value":{"type":"string","description":"The value of the thing. e.g. an identifier"},"type":{"type":"string"}}},"IdentifierType":{"title":"IdentifierType","type":"object","properties":{"id":{"type":"string"},"label":{"type":"string"},"type":{"type":"string"}},"description":"Relates a Identifier to a particular authoritative source identifier scheme: for example, if the identifier is MS.49 this property might indicate that this identifier has its origins in the Wellcome Library\'s CALM archive management system."},"Image":{"title":"Image","type":"object","description":"An image","properties":{"id":{"type":"string","description":"The canonical identifier given to a thing.","readOnly":true},"thumbnail":{"$ref":"#/components/schemas/DigitalLocation"},"locations":{"type":"array","description":"The locations which provide access to the image","items":{"$ref":"#/components/schemas/DigitalLocation"}},"source":{"description":"The entity from which an image was sourced","$ref":"#/components/schemas/Work"},"visuallySimilar":{"type":"array","description":"A list of visually similar images","items":{"$ref":"#/components/schemas/Image"}},"withSimilarColors":{"type":"array","description":"A list of images with similar color palettes","items":{"$ref":"#/components/schemas/Image"}},"withSimilarFeatures":{"type":"array","description":"A list of images with similar features","items":{"$ref":"#/components/schemas/Image"}},"type":{"type":"string"}}},"ImageAggregations":{"title":"ImageAggregations","type":"object","properties":{"source.genres.label":{"$ref":"#/components/schemas/Aggregation"},"source.contributors.agent.label":{"$ref":"#/components/schemas/Aggregation"},"locations.license":{"$ref":"#/components/schemas/Aggregation"},"type":{"type":"string"}},"description":"A map containing the requested aggregations."},"ImageResultList":{"title":"ImageResultList","type":"object","description":"A paginated list of images.","properties":{"type":{"type":"string"},"pageSize":{"type":"integer","format":"int32"},"totalPages":{"type":"integer","format":"int32"},"totalResults":{"type":"integer","format":"int32"},"results":{"type":"array","items":{"$ref":"#/components/schemas/Image"}},"prevPage":{"type":"string"},"nextPage":{"type":"string"},"aggregations":[{"$ref":"#/components/schemas/ImageAggregations"}]}},"Item":{"title":"Item","type":"object","description":"An item is a manifestation of a Work.","properties":{"id":{"type":"string","description":"The canonical identifier given to a thing."},"identifiers":{"type":"array","items":{"$ref":"#/components/schemas/Identifier"}},"title":{"type":"string","description":"A human readable title."},"note":{"type":"string","description":"Information to help distinguish different items."},"locations":{"type":"array","items":{"description":"A location that provides access to an item","discriminator":{"propertyName":"type"},"oneOf":[{"$ref":"#/components/schemas/DigitalLocation"},{"$ref":"#/components/schemas/PhysicalLocation"}]}},"status":{"$ref":"#/components/schemas/ItemStatus"},"type":{"type":"string"}}},"ItemStatus":{"title":"ItemStatus","type":"object","properties":{"id":{"type":"string"},"label":{"type":"string"},"type":{"type":"string"}},"description":"Item availability status"},"Language":{"title":"Language","type":"object","properties":{"id":{"type":"string","description":"An ISO 639-2 language code."},"label":{"type":"string","description":"The name of a language"},"type":{"type":"string"}},"description":"A language recognised as one of those in the ISO 639-2 language codes."},"License":{"title":"License","type":"object","properties":{"id":{"type":"string","description":"A type of license under which the work in question is released to the public.","enum":["cc-by","cc-by-nc","cc-by-nc-nd","cc-0, pdm"]},"label":{"type":"string","description":"The title or other short name of a license"},"url":{"type":"string","description":"URL to the full text of a license"},"type":{"type":"string"}},"description":"The specific license under which the work in question is released to the public - for example, one of the forms of Creative Commons - if it is a precise license to which a link can be made."},"LocationType":{"title":"LocationType","type":"object","properties":{"id":{"type":"string"},"label":{"type":"string"},"type":{"type":"string"}},"description":"The type of location that an item is accessible from."},"Note":{"title":"Note","type":"object","properties":{"contents":{"type":"array","items":{"type":"string","description":"The note contents."}},"noteType":{"$ref":"#/components/schemas/NoteType"},"type":{"type":"string"}},"description":"A note associated with the work."},"NoteType":{"title":"NoteType","type":"object","properties":{"id":{"type":"string"},"label":{"type":"string"},"type":{"type":"string"}},"description":"Indicates the type of note associated with the work."},"Period":{"title":"Period","type":"object","properties":{"id":{"type":"string"},"label":{"type":"string"},"identifiers":{"type":"array","items":{"$ref":"#/components/schemas/Identifier"}},"type":{"type":"string"}},"description":"A period of time"},"PhysicalLocation":{"title":"PhysicalLocation","type":"object","description":"A physical location that provides access to an item","properties":{"locationType":{"$ref":"#/components/schemas/LocationType"},"label":{"type":"string","description":"The title or other short name of the location."},"license":{"$ref":"#/components/schemas/License"},"shelfmark":{"type":"string","description":"The specific shelf where this item can be found"},"accessConditions":{"type":"array","items":{"$ref":"#/components/schemas/AccessCondition"}},"type":{"type":"string"}}},"Place":{"title":"Place","type":"object","properties":{"id":{"type":"string"},"label":{"type":"string"},"identifiers":{"type":"array","items":{"$ref":"#/components/schemas/Identifier"}},"type":{"type":"string"}},"description":"A place"},"ProductionEvent":{"title":"ProductionEvent","type":"object","description":"An event contributing to the production, publishing or distribution of a work.","properties":{"label":{"type":"string"},"places":{"type":"array","items":{"$ref":"#/components/schemas/Place"}},"agents":{"type":"array","items":{"$ref":"#/components/schemas/Agent"}},"dates":{"type":"array","items":{"$ref":"#/components/schemas/Period"}},"function":{"$ref":"#/components/schemas/Concept"},"type":{"type":"string"}}},"RelatedImage":{"title":"RelatedImage","type":"object","properties":{"id":{"type":"string","description":"The image ID"},"type":{"type":"string"}},"description":"An Image stub included on a work"},"RelatedWork":{"title":"RelatedWork","type":"object","description":"Stub for representing a work related to another work.","properties":{"id":{"type":"string","description":"The canonical identifier given to a thing.","readOnly":true},"title":{"type":"string","description":"The title or other short label of a work, including labels not present in the actual work or item but applied by the cataloguer for the purposes of search or description."},"referenceNumber":{"type":"string","description":"The identifier used by researchers to cite or refer to a work."},"partOf":{"type":"array","items":{"$ref":"#/components/schemas/RelatedWork"}},"totalParts":{"type":"integer","description":"Number of child works.","format":"int32"},"totalDescendentParts":{"type":"integer","description":"Number of descendent works.","format":"int32"},"type":{"type":"string"}}},"Subject":{"title":"Subject","type":"object","description":"A subject","properties":{"id":{"type":"string"},"identifiers":{"type":"array","items":{"$ref":"#/components/schemas/Identifier"}},"label":{"type":"string","description":"A label given to a thing."},"concepts":{"type":"array","items":{"discriminator":{"propertyName":"type"},"oneOf":[{"$ref":"#/components/schemas/Concept"},{"$ref":"#/components/schemas/Agent"},{"$ref":"#/components/schemas/Place"},{"$ref":"#/components/schemas/Period"}]}},"type":{"type":"string"}}},"Work":{"title":"Work","type":"object","description":"An individual work such as a text, archive item or picture; or a grouping of individual works (so, for instance, an archive collection counts as a work, as do all the series and individual files within it).  Each work may exist in multiple instances (e.g. copies of the same book).  N.B. this is not synonymous with \\\\\\"work\\\\\\" as that is understood in the International Federation of Library Associations and Institutions\' Functional Requirements for Bibliographic Records model (FRBR) but represents something lower down the FRBR hierarchy, namely manifestation. Groups of related items are also included as works because they have similar properties to the individual ones.","examples":[{}],"properties":{"id":{"type":"string","description":"The canonical identifier given to a thing.","readOnly":true},"title":{"type":"string","description":"The title or other short label of a work, including labels not present in the actual work or item but applied by the cataloguer for the purposes of search or description."},"alternativeTitles":{"type":"array","items":{"type":"string","description":"Alternative titles of the work."}},"referenceNumber":{"type":"string","description":"The identifier used by researchers to cite or refer to a work."},"description":{"type":"string","description":"A description given to a thing."},"physicalDescription":{"type":"string","description":"A description of specific physical characteristics of the work."},"workType":{"$ref":"#/components/schemas/Format"},"lettering":{"type":"string","description":"Recording written text on a (usually visual) work."},"createdDate":{"$ref":"#/components/schemas/Period"},"contributors":{"type":"array","description":"Relates a work to its author, compiler, editor, artist or other entity responsible for its coming into existence in the form that it has.","items":{"$ref":"#/components/schemas/Contributor"}},"identifiers":{"type":"array","items":{"$ref":"#/components/schemas/Identifier"}},"subjects":{"type":"array","items":{"$ref":"#/components/schemas/Subject"}},"genres":{"type":"array","items":{"$ref":"#/components/schemas/Genre"}},"thumbnail":{"$ref":"#/components/schemas/DigitalLocation"},"items":{"type":"array","items":{"$ref":"#/components/schemas/Item"}},"holdings":{"type":"array","items":{"$ref":"#/components/schemas/Holdings"}},"availabilities":{"type":"array","items":{"$ref":"#/components/schemas/Availability"}},"production":{"type":"array","items":{"$ref":"#/components/schemas/ProductionEvent"}},"languages":{"type":"array","items":{"$ref":"#/components/schemas/Language"}},"edition":{"type":"string","description":"Information relating to the edition of a work."},"notes":{"type":"array","items":{"$ref":"#/components/schemas/Note"}},"duration":{"type":"integer","description":"The playing time for audiovisual works, in seconds."},"images":{"type":"array","items":{"$ref":"#/components/schemas/RelatedImage"}},"parts":{"type":"array","description":"Child works.","items":{"$ref":"#/components/schemas/RelatedWork"}},"partOf":{"type":"array","description":"Ancestor works.","items":{"$ref":"#/components/schemas/RelatedWork"}},"precededBy":{"type":"array","description":"Sibling works earlier in a series.","items":{"$ref":"#/components/schemas/RelatedWork"}},"succeededBy":{"type":"array","description":"Sibling works later in a series.","items":{"$ref":"#/components/schemas/RelatedWork"}},"type":{"type":"string"}}},"WorkAggregations":{"title":"WorkAggregations","type":"object","properties":{"workType":{"$ref":"#/components/schemas/Aggregation"},"production.dates":{"$ref":"#/components/schemas/Aggregation"},"genres.label":{"$ref":"#/components/schemas/Aggregation"},"subjects.label":{"$ref":"#/components/schemas/Aggregation"},"contributors.agent.label":{"$ref":"#/components/schemas/Aggregation"},"languages":{"$ref":"#/components/schemas/Aggregation"},"items.locations.license":{"$ref":"#/components/schemas/Aggregation"},"availabilities":{"$ref":"#/components/schemas/Aggregation"},"type":{"type":"string"}},"description":"A map containing the requested aggregations."},"WorkResultList":{"title":"WorkResultList","type":"object","description":"A paginated list of works.","properties":{"type":{"type":"string"},"pageSize":{"type":"integer","format":"int32"},"totalPages":{"type":"integer","format":"int32"},"totalResults":{"type":"integer","format":"int32"},"results":{"type":"array","items":{"$ref":"#/components/schemas/Work"}},"prevPage":{"type":"string"},"nextPage":{"type":"string"},"aggregations":[{"$ref":"#/components/schemas/WorkAggregations"}]}}}},"tags":[{"name":"Images"},{"name":"Works"}]}}')}}]);